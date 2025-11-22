import torch
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--img_path", type=str, default='/mnt/sda/gyx/huawei_ad/data_self/rule_5/True/1149801842346982654_202507071417356ae22accfa3346769bab74fb5778af34.jpg', help="Path to the image to be evaluated")
    args = parser.parse_args()

    device = torch.device("cuda:3") if torch.cuda.is_available() else "cpu"

    # 从本地目录加载模型（替换为本地路径）
    model_path = 'QualiCLIP'
    model = torch.hub.load(repo_or_dir=model_path, source='local', model='QualiCLIP', pretrained=False)
    model.eval().to(device)
    
    # Path to the pre-trained weights file
    weights_path = "/home/gyx/huawei_ad/stage2/QualiCLIP-main/QualiCLIP+_koniq.pth"

    # Load the weights
    weights = torch.load(weights_path, map_location=device)

    # Load the weights into the model
    model.load_state_dict(weights, strict=False)
    
    # Define CLIP's normalization transform
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    # Load the image
    img = Image.open(args.img_path).convert("RGB")

    # Preprocess the images
    img = transforms.ToTensor()(img)
    img = normalize(img).unsqueeze(0).to(device)

    # Compute the quality score
    with torch.no_grad(), torch.cuda.amp.autocast():
        score = model(img)

    print(f"Image {args.img_path} quality score: {score.item()}")
