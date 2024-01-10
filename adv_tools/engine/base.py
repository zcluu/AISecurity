import abc
import numbers
from collections import defaultdict

from typing import Callable


class State(object):
    def __init__(self):
        self.iter = 0
        self.max_iter = None
        self.epoch_length = None
        self.dataloader = None
        self.seed = None

        self.metrics = dict()
        self.batch = None

    @property
    def current_epoch(self):
        if self.epoch_length is not None:
            return self.iter // self.epoch_length
        return None

    @property
    def max_epoch(self):
        if self.epoch_length is not None:
            return self.max_iter // self.epoch_length
        return None

    @property
    def current_batch_index(self):
        if self.epoch_length is not None:
            return self.iter % self.epoch_length
        return None

    @property
    def max_batch_index(self):
        return self.epoch_length

    def __repr__(self):
        rep = "State:\n"
        for attr, value in self.__dict__.items():
            if not isinstance(value, (numbers.Number, str, dict)):
                value = type(value)
            rep += "\t{}: {}\n".format(attr, value)
        return rep


class BaseEngine(abc.ABC):
    def __init__(self, logger, tb_writer):
        self.model = None
        self.logger = logger
        self.tb_writer = tb_writer
        self.stage = defaultdict(list)
        self.state = State()

    def add_hook_fn(self, stage, fn: Callable, **kwargs):
        self.stage[stage].append(fn)

    def setup(self, model):
        self.model = model

    def run(self, step_fn: Callable, dataloader, max_iter, start_iter=0, epoch_length=None, **kwargs):
        self.state.iter = self.state.start_iter = start_iter
        self.state.max_iter = max_iter
        self.state.epoch_length = epoch_length if epoch_length else len(dataloader)
        self.state.dataloader = dataloader
        self.state.dataloader_iter = iter(dataloader)
        self.state.step_fn = step_fn
        self.train(**kwargs)

    @abc.abstractmethod
    def train(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def step_fn(self, epoch):
        raise NotImplementedError
