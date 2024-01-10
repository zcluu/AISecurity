import os
import uuid

import math
import torch.nn
from typing import DefaultDict

from ..dist_utils.util import MetricLogger, TensorboardLogger


class BaselineTrainer(object):
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
        for epoch in range(epochs):
            self.sampler.get('train').set_epoch(epoch)
            header = 'Epoch: [{}]'.format(epoch)
            for step, batch in enumerate(self.dist_metric.log_every(self.train_loader, 10, header)):
                self.model.train()
                inputs, targets = batch
                inputs = inputs.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                self.optimizer.zero_grad()
                output = self.model(inputs)
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()

                self.dist_metric.update(loss=loss.item())
                if self.is_master:
                    if self.tb_writer:
                        self.tb_writer.update(head='loss', loss=loss.item())
                        self.tb_writer.set_step()

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
                correct += corr
            self.logger.info('Accuracy: {:.2f}%'.format(100 * correct / len(self.val_loader)))
