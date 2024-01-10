import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.transforms import transforms


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument("--data_dir", type=str, default='datasets')
    parser.add_argument("--weight", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = make_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    datasets = ImageFolder(args.data_dir, transform=transform)
    dataloader = DataLoader(datasets, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    cls_num = len(os.listdir(args.data_dir))
    model = resnet50(num_classes=cls_num)
    model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'])
    model.to(device)
    tot_acc = 0
    with torch.no_grad():
        model.eval()
        for data in dataloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct = predicted.eq(labels.data)
            accuracy = correct.sum().item() / labels.size(0)
            tot_acc += accuracy
    print(tot_acc / len(dataloader))


if __name__ == '__main__':
    main()
