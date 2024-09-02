import os
import math
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from model import resnet34, resnet101
from my_dataset import MyDataSet
from utils import read_split_data
from multi_train_utils.train_eval_utils import train_one_epoch, evaluate

###注意这里的数据集传的是人家电脑的路径，记得要改成自己存放花朵数据集的路径

#首先先定义传入的参数配置
def main(args):      #我们给主函数中传入配置参数的对象
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#显示训练设备，对于device我们通常用torch.device传入字符串，且按上面那么写（判断语句的简写,首先先判断电脑中有没有装gpu）
#传入的内容为字符串，"cuda:0" / "cuda:1" / "cpu"

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()                 #生成tensorboard对象，用来记录训练过程中的损失，测试集上表现的准确率，学习率的变化曲线
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    #训练后我们要把模型权重保存在这个文件夹内

    train_info, val_info, num_classes = read_split_data(args.data_path) #
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info
     #这样就将训练集和验证集进行了划分，分别得到了样本地址的列表，并按样本的排列顺序分别得到了训练和验证集对应的标签

    # check num_classes
    assert args.num_classes == num_classes, "dataset num_classes: {}, input {}".format(args.num_classes,
                                                                                       num_classes)
    
    #开始进行数据预处理
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

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw)) #表示选择cpu的数量
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # 如果存在预训练权重则载入
    model = resnet34(num_classes=args.num_classes).to(device) #把模型复制到device上
    

    # if os.path.exists(args.weights):
    #      weights_dict = torch.load(args.weights, map_location=device)
    #      load_weights_dict = {k: v for k, v in weights_dict.items()
    #                          if model.state_dict()[k].numel() == v.numel()}
    #      model.load_state_dict(load_weights_dict, strict=False)
    
    #上面是使用预训练权重的代码，这里我们就不使用了

    # 是否冻结权重
    # if args.freeze_layers:
    #     for name, para in model.named_parameters():
    #         # 除最后的全连接层外，其他权重全部冻结
    #         if "fc" not in name:
    #             para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    #将所有需要优化的参数打包成列表送入到优化器中
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.epochs):
        # train
        mean_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        scheduler.step()

        # validate
        # sum_num = evaluate(model=model,
        #                    data_loader=val_loader,
        #                    device=device)
        # acc = sum_num / len(val_data_set)
        # print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))
        # tags = ["loss", "accuracy", "learning_rate"]
        # tb_writer.add_scalar(tags[0], mean_loss, epoch)
        # tb_writer.add_scalar(tags[1], acc, epoch)
        # tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

        # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


#训练文件唯一面向过程的地方，解析命令行参数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  #创建解析对象
    parser.add_argument('--num_classes', type=int, default=5) #如果不写type 默认传入的是字符串类型
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./flower_data/flower_photos")
    

    

    # resnet34 官方权重下载地址
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resNet34.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    #'cuda'默认是'cuda:0'   
                                  
    opt = parser.parse_args()   # 最终汇总返回

    main(opt)
