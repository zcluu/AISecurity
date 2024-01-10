import argparse
import os
import sys
from collections import defaultdict

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from cv_lib.utils import make_deterministic
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms

sys.path.append(os.getcwd())

from adv_tools.engine.baseline import BaselineTrainer
from adv_tools.dist_utils.util import TensorboardLogger
from adv_tools.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument("--data_dir", type=str, default='/home/lzc/SourceCode/AISecurity/Adversarial Attack/datasets')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("-distributed", action="store_true")
    parser.add_argument('--ip', default='127.0.0.1', type=str)
    parser.add_argument('--port', default='23456', type=str)
    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    mp.spawn(main_worker, nprocs=args.nprocs, args=(args.nprocs, args))


def main_worker(local_rank, nprocs, args):
    logger = get_logger(name='ResNet50')
    args.local_rank = local_rank
    make_deterministic(args.seed)
    init_method = 'tcp://' + args.ip + ':' + args.port
    # Init Distribute
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method=init_method, world_size=args.nprocs, rank=local_rank)
    # Init Model
    cls_num = len(os.listdir(os.path.join(args.data_dir, 'train')))
    logger.info(f'Number of classes: {cls_num}')
    model = resnet50(num_classes=cls_num)
    ckpt = torch.load('/mnt/weights/resnet50-0676ba61.pth')
    del ckpt['fc.weight']
    del ckpt['fc.bias']
    model.load_state_dict(ckpt, strict=False)
    torch.cuda.set_device(local_rank)
    model.cuda(local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    logger.info('Model load successfully...')
    batch_size = int(args.batch_size / nprocs)

    transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ImageFolder(
        root=os.path.join(args.data_dir, 'train'),
        transform=transform,
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=nprocs, rank=local_rank, shuffle=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    lr = 3e-4
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    if args.local_rank == 0:
        tb_writer = TensorboardLogger(log_dir=args.save_dir)
    else:
        tb_writer = None
    dataloader = defaultdict(None)
    dataloader['train'] = train_loader
    trainer = BaselineTrainer(
        logger, tb_writer, dataloader,
        model, optimizer, loss_fn, {'train': train_sampler},
        is_master=args.local_rank == 0,
        save_dir=args.save_dir,
        lr=lr
    )
    trainer.train(100)


if __name__ == '__main__':
    main()
