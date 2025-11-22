# coding=utf-8

from __future__ import absolute_import, division, print_function
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data import Dataset
import cv2

class NumpyImageDataset(Dataset):
    """Custom dataset that accepts numpy arrays and returns image paths alongside images and labels"""
    def __init__(self, images,  transform=None):
        """
        Args:
            images (numpy.ndarray): Array of images in numpy format (N, H, W, C)
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


def is_realistic_or_postprocessing(ViT_model,img_np):

#    print(ViT_model)

    transform_test = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)  # shape: (H, W, C), RGB格式
    
    # 2. 添加batch维度 (N=1)
    img_np = np.expand_dims(img_rgb, axis=0)  # shape: (1, H, W, C)
    
    numpy_images = img_np  # shape (N, H, W, C)
    testset = NumpyImageDataset(numpy_images, transform=transform_test) 
    test_sampler = SequentialSampler(testset)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=1,
                             num_workers=1,
                             pin_memory=True) if testset is not None else None
    ViT_model.zero_grad()
    ViT_model.eval()
    for step, batch in enumerate(test_loader):
        # Now batch contains (images, labels, paths)
        images= batch.cuda()

        with torch.no_grad():
            logits = ViT_model(images)[0]
            preds = torch.argmax(logits, dim=-1)
    
    if preds==1:
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
