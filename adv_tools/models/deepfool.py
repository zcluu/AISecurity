import numpy as np
import torch

from .adv_base import AdvModel


class DeepFool(AdvModel):
    def __call__(self, model, x):
        adv_x = x.clone().requires_grad_()
        adv_logits = model(adv_x)
        adv_labels = adv_logits.argmax(dim=1)
        if adv_labels.size() == ():
            adv_labels = torch.tensor([adv_labels])
        ori_labels = adv_labels

        w = torch.squeeze(torch.zeros(x.size()[1:])).to(self.device)
        r_tot = torch.zeros(x.size()).to(self.device)

        cur_iter = 0
        while (adv_labels == ori_labels).any and cur_iter < self.max_iter:
            pred = adv_logits.topk(self.candidate)[0]
            gradients = torch.stack(self.jacobian(pred, adv_x, self.candidate), dim=1)
            with torch.no_grad():
                for idx in range(x.size(0)):
                    pert = np.inf
                    if adv_labels[idx] != ori_labels[idx]:
                        continue
                    for k in range(1, self.candidate):
                        w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                        f_k = pred[idx, k] - pred[idx, 0]
                        pert_k = (f_k.abs() + 0.00001) / w_k.view(-1).norm()
                        if pert_k < pert:
                            pert = pert_k
                            w = w_k

                    r_i = pert * w / w.view(-1).norm()
                    r_tot[idx, ...] = r_tot[idx, ...] + r_i

            adv_x = ((1 + self.overshoot) * r_tot + x).requires_grad_()
            adv_logits = model(adv_x)
            adv_labels = adv_logits.argmax(dim=1)
            if adv_labels.size() == ():
                adv_labels = torch.tensor([adv_labels])
            cur_iter = cur_iter + 1
        print('cur_iter', cur_iter)
        adv_x = (1 + self.overshoot) * r_tot + x
        return adv_x
