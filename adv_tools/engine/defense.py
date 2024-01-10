import os
import uuid

import math
import torch.nn
from typing import DefaultDict

from ..dist_utils.util import MetricLogger, TensorboardLogger
from ..utils.logger import get_logger


def l2_loss(x, y):
    diff = x - y
    diff = diff * diff
    diff = diff.sum(1)
    diff = diff.mean(0)
    return diff


class DefenseTrainer(object):
    def __init__(
            self,
            logger,
            tb_writer: TensorboardLogger,
            dataloader: DefaultDict,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            loss_fn,
            sampler,
            is_master=False,
            save_dir=f'results/{uuid.uuid4()}',
            lr=1e-1
    ):
        self.logger = logger
        self.tb_writer = tb_writer
        self.is_master = is_master
        self.dataloader = dataloader
        self.train_loader = dataloader.get('train', None)
        self.val_loader = dataloader.get('val', None)
        self.test_loader = dataloader.get('test', None)
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.dist_metric = MetricLogger(is_master=is_master, logger=logger)
        self.save_dir = save_dir
        self.lr = lr

    def train(self, epochs):
        p_val = 5
        l_ce = 1
        l_reg = 1
        mul_ru = 9
        qnoise_scale = math.pow(2, min(p_val - 2, 3))
        for epoch in range(epochs):
            self.sampler.get('train').set_epoch(epoch)
            header = 'Epoch: [{}]'.format(epoch)
            for step, batch in enumerate(self.dist_metric.log_every(self.train_loader, 10, header)):
                self.model.train()
                inputs, targets = batch
                B, C, H, W = inputs.size()
                pow_p = torch.ones(B, C, H, W).cuda() * math.pow(2, p_val)
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                qdata = torch.round(inputs * 255)
                qnoise = torch.Tensor(inputs.size()).uniform_(-1, 1).cuda() * qnoise_scale
                qdata = qdata + qnoise
                qdata = qdata - (qdata % pow_p) + (pow_p / 2)
                qdata = torch.clamp(qdata, 0, 255)
                qdata = qdata / 255

                self.optimizer.zero_grad()
                q_out = self.model(qdata)
                ori_out = self.model(inputs)

                reg = l2_loss(ori_out, q_out)
                closs = self.loss_fn(ori_out, targets)
                cost = l_ce * closs + l_reg * reg
                cost.backward()
                self.optimizer.step()

                self.dist_metric.update(reg=reg.item())
                self.dist_metric.update(closs=closs.item())
                self.dist_metric.update(cost=cost.item())
                if self.is_master:
                    if self.tb_writer:
                        self.tb_writer.update(head='loss', reg=reg.item())
                        self.tb_writer.update(head='loss', closs=closs.item())
                        self.tb_writer.update(head='loss', cost=cost.item())
                        self.tb_writer.set_step()

            # if epoch == 25:
            #     l_reg = l_reg * mul_ru
            #     # for param_group in self.optimizer.param_groups:
            #     #     param_group['lr'] = self.lr / 5
            # elif epoch == 50:
            #     l_reg = l_reg * mul_ru
            # #                 for param_group in self.optimizer.param_groups:
            # #                     param_group['lr'] = self.lr / 25
            # elif epoch == 75:
            #     l_reg = l_reg * mul_ru
            #                 for param_group in self.optimizer.param_groups:
            #                     param_group['lr'] = self.lr / 125

            if self.is_master:
                torch.save({
                    'model': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, os.path.join(self.save_dir, '{:04d}.pth'.format(epoch)))

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            for batch_ix, data in enumerate(self.val_loader):
                images, labels = data
                images = images.to('cuda')
                labels = labels.to('cuda')
                output = self.model(images)
                prediction = output.argmax(dim=1, keepdim=True)
                corr = torch.sum(prediction == labels).item() / images.size(0)
                self.logger.info(f'Acc: {round(corr, 4)}')
