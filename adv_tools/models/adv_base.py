import torch
import abc


class AdvModel(abc.ABC):
    def __init__(
            self,
            candidate: int = 100,
            overshoot: float = 0.02,
            max_iter: int = 50,
            clip_min: float = 0,
            clip_max: float = 1,
            device='cuda'
    ):
        self.candidate = candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.device = device

    @abc.abstractmethod
    def __call__(self, model, x):
        raise NotImplementedError

    @staticmethod
    def jacobian(predictions, x, nb_classes):
        list_derivatives = []
        for cls_ix in range(nb_classes):
            outputs = predictions[:, cls_ix]
            derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
            list_derivatives.append(derivatives)

        return list_derivatives
