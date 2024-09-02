
import os
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl') 
rank=int(os.environ['RANK'])
world_size=int(os.environ['WORLD_SIZE'])
device=torch.device(f'cuda:{rank}')
tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
li=[torch.zeros_like(tensor) for i in range(world_size)]
dist.all_gather(li,tensor)
if rank == 0:
    print(li)


#当你将一个张量存储在列表中时，列表会保存对该张量的引用，而不是仅仅保存标签或其他信息。因此，如果你的张量最初是在 GPU 上创建的，那么当你将它存储在列表中时，它依然会保留在 GPU 上。
#dist.all_gather和dist.all_reduce都是在每个进程都进行了对于所有进程相关张量的汇总，产生的张量的分别还是在各自进程上
a=round(3.125,2)
