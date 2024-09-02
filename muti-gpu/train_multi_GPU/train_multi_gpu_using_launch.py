import os
import math
import tempfile
import argparse
 
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from model import resnet34
from my_dataset import MyDataSet
from utils import read_split_data, plot_data_loader_image
from multi_train_utils.distributed_utils import init_distributed_mode, dist, cleanup
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate


def main(args):
    if torch.cuda.is_available() is False:       ##判断电脑里是否有cuda
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args) #将各自进程与其它进程建立了通信，并得到了自己进程gpu的索引

    rank = args.rank                 #在自己gpu上训练，先需要初始化参数，这时候就可以看作是单卡训练
    device = torch.device(args.device) #对于自己所在进程，根据自己gpu的索引配置设备信息
    batch_size = args.batch_size
    weights_path = args.weights
    args.lr *= args.world_size  # 学习率要根据并行GPU的数量进行倍增，因为采用gpu并行相当于更新一次梯度的真正batch_size是单gpu的k倍（k为gpu的个数），更新的次数少，一次需要下降更多梯度，所以学习率大
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard（一般画图，打印信息，保存模型等操作在一个进程中进行即可）
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

    train_info, val_info, num_classes = read_split_data(args.data_path)
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info

    # check num_classes
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])

    # 实例化验证数据集
    val_data_set = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])
    #为了保证每个进程都能独立处理数据，所以数据的准备工作也即dataset的创建也是在每个进程上都进行

    # 给每个rank对应的进程分配训练的样本索引 （先分配数据）每个gpu获得自己的总数据集
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)

    # 将样本索引每batch_size个元素组成一个list，先组织好一个个batch，后面送入dataloader进行一个个取出也即采样
    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sampler, batch_size, drop_last=True)
      
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #用来获得cpu进行数据预处理的线程数，采用多线程加速数据计算

    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             sampler=val_sampler,        #这里和训练不一样，因为验证不需要小批量了，一般是遍历整个数据集
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)
    # 实例化模型
    model = resnet34(num_classes=num_classes).to(device)

    # 如果存在预训练权重则载入
    # if os.path.exists(weights_path):
    #     weights_dict = torch.load(weights_path, map_location=device)
    #     load_weights_dict = {k: v for k, v in weights_dict.items()
    #                          if model.state_dict()[k].numel() == v.numel()}
    #     model.load_state_dict(load_weights_dict, strict=False)
    
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
        # 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致(因为正常模型在初始化时初始权重是随机的，所以不同进程是不同的)
    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)

    dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        #先保存第一个进程的权重，后续进程再加载这个权重文件，必须保证一开始所有进程上model的初始化权重都相同

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)
    else:
        # 只有训练带有BN结构的网络时使用SyncBatchNorm采用意义
        if args.syncBN:
            # 使用SyncBatchNorm后训练会更耗时
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
            #使用具有同步功能的BN层
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #在分布式训练中，每个进程（通常每个进程对应一个 GPU）都会计算梯度。DistributedDataParallel 会自动处理梯度的同步，使得每个进程的梯度在进行参数更新之前都能保持一致。

    # optimizer
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)  #不同的epoch每个gpu分配到的数据是不同的，内置了随机种子

        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        sum_num = evaluate(model=model,
                           data_loader=val_loader,
                           device=device)
        acc = sum_num / val_sampler.total_size

        if rank == 0:
            print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup() #释放资源


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)
    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default="./flower_data/flower_photos")

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')#可以简单理解为是一种通讯协议和地址，是为建立多进程间通信而引入的，和args.dist_backend = 'nccl'一起帮助建立通信
    opt = parser.parse_args()

    main(opt)
