import argparse
import os
import sys
from pathlib import Path

from torchvision.datasets.folder import default_loader

import torch
from torchvision.models.resnet import resnet50
from torchvision import transforms

ROOT = Path(os.getcwd())
sys.path.append(os.getcwd())

from adv_tools.models.deepfool import DeepFool
from adv_tools.utils.logger import get_logger


def make_args():
    parser = argparse.ArgumentParser()
    # Image Config
    parser.add_argument('--img_size', type=int, default=224)
    # Train Config
    parser.add_argument('--img_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=ROOT / 'results')
    parser.add_argument('--origin_weight', type=str)
    parser.add_argument('--attack_weight', type=str)
    # Model Config
    parser.add_argument('--cls_num', type=int, default=10)

    return parser


def main():
    logger = get_logger('attackAttack')
    args = make_args().parse_args()
    logger.info('Args: {}'.format(args))

    ori_model = resnet50(num_classes=10).cuda()
    def_model = resnet50(num_classes=10).cuda()
    ori_model.load_state_dict(torch.load(args.origin_weight)['model'], strict=True)
    def_model.load_state_dict(torch.load(args.attack_weight)['model'], strict=True)
    ori_model.eval()
    def_model.eval()

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    filename = os.path.basename(args.img_dir).split('.')[0]
    image = default_loader(args.img_dir)
    image = transform(image)
    image = image.unsqueeze(0).cuda()
    attack = DeepFool(candidate=10, max_iter=100)
    attack_x = attack(ori_model, image)

    with torch.no_grad():
        ori_logit = ori_model(image).argmax(dim=1)
        attack_logit = ori_model(attack_x).argmax(dim=1)
        def_logit = def_model(attack_x).argmax(dim=1)
        print('Attack Result: ', ori_logit == attack_logit)
        print('Defense Result: ', ori_logit == def_logit)
        print('Original Category: ', ori_logit.item())
        print('Attacked Category: ', attack_logit.item())
        print('Defense Category: ', def_logit.item())


if __name__ == '__main__':
    main()
