import os
import sys
from tqdm import tqdm
import torch
import torch.distributed as dist
import argparse
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from model import resnet34
from my_dataset import MyDataSet
from utils import read_split_data,plot_data_loader_image
from torch.utils.data import DataLoader
import math


def init_args():
    """ 
    用于解析命令行参数    
    """
    parser = argparse.ArgumentParser(description='multi-gpu-flowers-classification')                                                    
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)                                       #初始学习率
    parser.add_argument('--lrf', type=float, default=0.1)                                        #最终学习率 
    parser.add_argument('--data_path', type=str, default="./flower_data/flower_photos")          #数据集的地址
    args = parser.parse_args()
    return args                                                                                  #生成配置对象



def init_distribute(args):                                  #其实这里不用创建变量args.xxx，直接创建正常变量即可，因为这里的参数都不是从外部传入的，而就是普通的局部变量，但我们为了形式的整齐就先这么写了，如果真的有需要可以从命令行进行修改
    """
    用于初始化进程组，建立多进程通讯
    """
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.") #先来判断环境是否装有cuda
    args.rank=int(os.environ['RANK'])                               #os.environ返回的环境变量是str，需要转为int
    args.world_size = int(os.environ['WORLD_SIZE'])                 #在一机多卡中，'WORLD_SIZE'表示进程总数，也即GPU总数，命令行中传入的nproc_per_node为多少，那么'WORLD_SIZE'就为多少
                                                                    #'RANK'表示这是哪一个进程，也即指向GPU的索引标号，是第几个GPU  
    args.dist_backend = 'nccl'                                      #设置通信后端，为后面建立不同进程间的通讯做准备。
    args.dist_url='env://'                                          #表示一种初始化分布式训练的方式，这里'env://' 指的是从环境变量中读取分布式训练的初始化信息。（也就是上述环境变量os.environ那些）
    args.distributed = True                                         #标记这个程序启动了分布式训练，一般在多卡和单卡混合代码时常常把这个作为判断标识
    
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,  #初始化进程组，简单来说就是建立多GPU间的一个通讯，为后面的计算做准备
                            world_size=args.world_size, rank=args.rank)                       
    dist.barrier()                                                 #确保所有进程在此处同步，跑的快的进程要在这里等跑的慢的进程

def train_one_epoch(model, optimizer, train_dataloader, device, epoch,loss,rank,world_size):
    model.train()                                                  #启动训练模式，意味着相应BN，dropout层会做出相应反应，因为它们在训练和验证时具有不同的操作
    mean_loss = torch.zeros(1).to(device)                          #先赋一个初值，用于后续多轮次平均损失的求解
    optimizer.zero_grad()                                          #再每个epoch循环前，先清空优化器内储存的梯度信息
    if rank == 0 :
        train_dataloader = tqdm(train_dataloader, file=sys.stdout) #在主程序中用来生成训练的进度条
    for step, data in enumerate(train_dataloader):                 #提取索引，数据 step即一个epoch中取出的批次，用来计算一个epoch的平均损失  
        images, labels = data
        images=images.to(device)                                   #把数据移动到gpu上
        labels=labels.to(device)
        l=loss(model(images),labels)
        l.backward()                                               #梯度反向传播，这里的梯度信息已经被储存在了模型的参数中，因此只要把相应权重参数送入优化器中就能进行梯度更新
        optimizer.step()                                           #更新权重参数
        optimizer.zero_grad()                                      #双保险，前后各清空一次梯度
        #现在我们希望计算每个epoch中每个step的损失函数，并随着step的迭代它也跟着迭代，是该epoch中结束的step的平均值，比较能代表此轮epoch的真实损失值大小
        with torch.no_grad():                                      #一般对于损失函数进行额外计算时，都要保证其设置为不求解梯度，即不参与计算图的构建
            dist.all_reduce(l)                                     #计算多个gpu上的平均损失，才是真正该step的损失。这里注意没有返回，是直接把求和的值存在了dist.all_reduce的括号中的值。相当于每个进程都获得了该值的总和
            l=l / world_size                                       #计算均值，记得除去进程数
        mean_loss = (mean_loss * step + l.detach()) / (step + 1)   #计算该epoch中已经计算完的step的平均损失，一定要注意所有数据都得在gpu上！！！
                                                                   #两种方法都可以确保梯度计算被禁用，但 detach() 是对单个张量的操作，而 torch.no_grad() 是对整个代码块的操作，一个是张量在运算这一步时不参与梯度图构建，另一个是整个代码块都不参与梯度计算
        if rank == 0:
            train_dataloader.desc = "[epoch {}] mean loss {}".format(epoch, round(mean_loss.item(), 3)) #也可以用f''的字符串插入数值写法,这个就跟上面的那个进度条一般一起使用，在每个epoch打印进度时不断更新它的mean_loss

    torch.cuda.synchronize(device)                                 #一般加在训练函数和验证函数的最后，确保gpu上所有的计算结束
    return mean_loss.item()                                        #返回标量值


def main():
    args=init_args()
    init_distribute(args)
    epochs=args.epochs
    batch_size = args.batch_size
    data_path=args.data_path
    lr=args.lr
    lrf=args.lrf
    world_size=args.world_size                    
    rank=args.rank                                #读取命令行参数和gpu序号rank
    device = torch.device(f"cuda:{rank}")         #创建每个进程对应的设备信息


    # torch.cuda.set_device 对应 torch.randn(10).cuda()
    # device = torch.device(args.device if torch.cuda.is_available() else "cpu") 对应 torch.randn(10).to(device)
    # 上述两种方法都可以先指定设备，再把模型和数据移到相应设备（GPU）上进行训练


    if rank == 0:                                  #一般tensorboard画图，打印配置信息与模型信息，保存模型等操作在一个进程中进行即可，一般默认0为主进程，就在主进程中打印即可
        print(args)    
        if os.path.exists("./weights") is False:   #创建模型保存的文件
            os.makedirs("./weights")              
    

    train_info, val_info, num_classes = read_split_data(data_path)      #这边是已经提前写好的代码用来处理原始数据，划分数据集，生成训练集验证集的列表以及其对应的标签文件
    train_images_path, train_images_label = train_info
    val_images_path, val_images_label = val_info


    data_transform = {                                                     #数据预处理，生成dataset 
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose  ([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
    
    train_dataset = MyDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=data_transform["train"])
    val_dataset = MyDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=data_transform["val"])  
    


    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    #用来获得cpu进行数据预处理的线程数，采用多线程加速数据计算,在dataloader中会使用


    #分布式训练数据的加载
    #DistributedSampler把所有数据分成N份（N为进程数也即使用的GPU的个数），之后将其能分给N个GPU进行训练。相当于在每个epoch中，每个GPU得到总样本的1/N，并以此作为自己训练样本的总体，送入dataloader，按照正常那样一批量一批量取出计算，直到遍历完一遍则一个epoch结束
    #这里如果无法均匀分配则会重复复制一些数据让它总数能被N整除
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    #上述得到了每个gpu分到的总样本，接下来把它送进各自的dataloader中
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=nw)

    model = resnet34(num_classes=num_classes).to(device)                                        #初始化模型
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)                     #对于BN层采用在这里统一计算所有gpu上样本的均值和方差，在此处快的gpu会等待慢的gpu，进度实现一个统一
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank]).to(device)       # 转为DDP模型,也即建立不同gpu上模型的关联，为后面优化综合所有gpu的信息做准备。转为DDP模型后，pytorch内部已经保证了各进程上model的初始化权重是相同的。
    #if rank == 0:                                         
        #print(model)              #打印模型信息
    
    dist.barrier()                #加载完模型就准备开始先训练了，先让大家统一下进度

    #加载损失函数
    loss = torch.nn.CrossEntropyLoss() #类的初始化

    #加载优化器
    pg = [p for p in model.parameters() if p.requires_grad]                   #遍历model中所有可求梯度的参数，送入优化器内
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=0.005)        #优化器只要你给我梯度信息，我就负责对参数进行梯度下降即可。而梯度信息则由损失函数求backward而来
    
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # 采用一种随着训练不断更新学习率的算法
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    

    #开始训练和验证过程
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)  #保证了每个epoch分到各个gpu的样本是不同的
        mean_loss = train_one_epoch(model,optimizer,train_dataloader,device,epoch,loss,rank,world_size)
        scheduler.step()                #更新优化器中的学习率，随epoch变化

       #每个epoch结束前做一次验证，再封装一个验证函数，也是通过生成dalaloader，每次一整个批量进行验证，遍历完所有数据即可，只需要一个epoch，记得设置不计算梯度！！！
    
    dist.destroy_process_group() #销毁分布式进程组，收尾呼应

if __name__ == '__main__':
    main()
