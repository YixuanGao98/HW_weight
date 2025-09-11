# coding=utf-8

from __future__ import absolute_import, division, print_function
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse

import random
import numpy as np

from PIL import Image
import torch
from models.modeling import VisionTransformer, CONFIGS
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import Dataset

import cv2
class NumpyImageDataset(Dataset):
    """Custom dataset that accepts numpy arrays and returns image paths alongside images and labels"""
    def __init__(self, images, labels=None, image_paths=None, transform=None):
        """
        Args:
            images (numpy.ndarray): Array of images in numpy format (N, H, W, C)
            labels (numpy.ndarray): Array of labels (optional)
            image_paths (list): List of image paths (optional)
            transform (callable, optional): Optional transform to be applied
        """
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]  # Get the specific image at this index
        
        # Convert numpy array to PIL Image
        if len(image.shape) == 3:  # (H, W, C)
            image = Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image
def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    numpy_images = args.img_np  # shape (N, H, W, C)
    testset = NumpyImageDataset(numpy_images, transform=transform_test) if args.local_rank in [-1, 0] else None
    if args.local_rank == 0:
        torch.distributed.barrier()

    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=1,
                             pin_memory=True) if testset is not None else None

    return test_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup(args):
    # Prepare model
    config = CONFIGS['R50-ViT-B_16']
    num_classes = 100
    model = VisionTransformer(config, 384, zero_head=True, num_classes=num_classes)
    
    # Load weights from PyTorch checkpoint (.bin, .pth)
    if args.pretrained_dir:
        checkpoint = torch.load(args.pretrained_dir)

        # Handle different checkpoint formats
        if "state_dict" in checkpoint:  # Common in PyTorch Lightning
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:     # Some checkpoints use "model"
            state_dict = checkpoint["model"]
        else:                           # Direct state_dict
            state_dict = checkpoint

        # Remove "module." prefix if present (from DataParallel/DDP)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load weights into the model
        model.load_state_dict(state_dict, strict=True)
    
    model.to(args.device)
    num_params = count_parameters(model)
    # print(num_params)
    return args, model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

import pandas as pd
def valid(args, model, test_loader):
    # Validation!


    model.eval()
    for step, batch in enumerate(test_loader):
        # Now batch contains (images, labels, paths)
        images= batch.to(args.device)
        
        # batch = tuple(t.to(args.device) for t in batch)
        # x, y = batch
        with torch.no_grad():
            logits = model(images)[0]

            preds = torch.argmax(logits, dim=-1)

    return preds


def train(args, model):
    """ Train the model """
    test_loader = get_loader(args)
    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    preds=valid(args, model, test_loader)
    return preds

def is_realistic_or_postprocessing(pretrained_dir,img_np):
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--pretrained_dir", type=str, default="/DATA/DATA1/gyx/huawei_ad/model/ViT/hw_checkpoint.bin",
                        help="Where to search for pretrained ViT models.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--img_np", type=int, default=None,
                        help="img_npy")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")


    args = parser.parse_args()
    args.pretrained_dir=pretrained_dir
    # 1. 转换通道顺序: BGR -> RGB（如果模型需要RGB输入）
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # shape: (H, W, C), RGB格式
    
    # 2. 添加batch维度 (N=1)
    img_np = np.expand_dims(img_rgb, axis=0)  # shape: (1, H, W, C)
    

    
    args.img_np=img_np
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)
    preds=train(args, model)
    if preds==0:
        return True
    else:
        return False


if __name__ == "__main__":
    pretrained_dir="/DATA/DATA1/gyx/huawei_ad/model/ViT/hw_checkpoint.bin"
    # 1. 读取图像
    img_path = "/DATA/DATA1/gyx/huawei_ad/objective_database/test/真实感/负样本/IMG_9031.jpg"  # 替换为你的图像路径
    img = Image.open(img_path)  # 用PIL读取图像

    # 2. 转换为NumPy数组 (H, W, C)
    img_np = np.array(img)  # shape: (height, width, channels), RGB格式
    # Add batch dimension (N=1 for single image)
    img_np = np.expand_dims(img_np, axis=0)  # shape becomes (1, H, W, C)
    preds=main(pretrained_dir,img_np)
    # print(preds)