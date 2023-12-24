import argparse
import sys, os
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from deepfool import DeepFool
from functional import resume_img
from logger import get_logger

ROOT = Path(os.getcwd())

import torch
import torchvision.transforms.functional as TF
from torchvision.models.resnet import resnet50
from torchvision import transforms
from torchvision.datasets import ImageFolder


def make_args():
    parser = argparse.ArgumentParser()
    # Image Config
    parser.add_argument('--img_size', type=int, default=224)
    # Train Config
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_dir', type=str, default=ROOT / 'results')
    parser.add_argument('--batch_size', type=int, default=128)
    # Model Config
    parser.add_argument('--cls_num', type=int, default=1000)

    return parser


def main():
    logger = get_logger('AdvAttack')
    args = make_args().parse_args()
    logger.info('Args: {}'.format(args))
    model = resnet50(pretrained=True).cuda()
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10, pin_memory=True)

    attack = DeepFool(candidate=10, max_iter=100)
    dic = {
        'ori_correct': [],
        'adv_correct': [],
    }
    for step, (images, labels) in enumerate(dataloader):
        images = images.to('cuda')
        labels = labels.to('cuda')
        adv_x = attack(model, images)
        with torch.no_grad():
            ori_logits = model(images).argmax(dim=1)
            adv_logits = model(adv_x).argmax(dim=1)
        ori_correct = torch.sum(ori_logits == labels).item() / images.size(0)
        adv_correct = torch.sum(adv_logits == labels).item() / images.size(0)
        dic['ori_correct'].append(ori_correct)
        dic['adv_correct'].append(adv_correct)
        pd.DataFrame(dic).to_csv('result.csv')
        logger.info(
            f'Step: {step}, Ori Acc: {round(ori_correct, 4)}, Adv Acc: {round(adv_correct, 4)}'
        )
        if step == 0:
            os.makedirs(os.path.join(f'{args.save_dir}/{step}'), exist_ok=True)
            for i in range(adv_x.size(0)):
                ori_img = resume_img(images[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                adv_img = resume_img(adv_x[i], [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                TF.to_pil_image(ori_img).save('{}/{}/{:03d}_ori.jpg'.format(
                    args.save_dir, step, i
                ))
                TF.to_pil_image(adv_img).save('{}/{}/{:03d}_adv.jpg'.format(
                    args.save_dir, step, i
                ))


if __name__ == '__main__':
    main()
