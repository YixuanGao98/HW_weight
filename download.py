import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# from huggingface_hub import snapshot_download

# snapshot_download(repo_id='laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup',
#                   repo_type='model', local_dir = '/mnt/sda/gyx/huggingface/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup',
#                   resume_download=True,force_download=True)
# # #
# snapshot_download(repo_id='liuhaotian/LLaVA-Lightning-7B-delta-v1-1',
#                   repo_type='model', local_dir = '/DATA/DATA1/gyx/checkpoint/llava/LLaVA-Lightning-7B-delta-v1-1',
#                   resume_download=True,force_download=True)

# snapshot_download(repo_id='TencentARC/SmartEdit-7B',
#                 repo_type='dataset', local_dir = r'C:\Users\13356\Desktop\underwater\saliency\MMVP',
#                 resume_download=True)
# 模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('thunderbolt/ArtiMuse', local_dir = '/mnt/DATA2025/gyx/checkpoint/ArtiMuse')
# model_dir = snapshot_download('thunderbolt/ArtiMuse_AVA', local_dir = '/mnt/DATA2025/gyx/checkpoint/ArtiMuse_AVA')

# # 下载数据
#snapshot_download(repo_id = 'Iceclear/AVA',  
#                   repo_type="dataset",  # 可选 [dataset,model] 
#                    local_dir='/DATA/DATA1/gyx/Image_database/AVA',# 下载到本地的路径
#                      resume_download=True)  # 自己的hf token 不是必须的，仅有一些hub需要