
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

Could not fetch URL https://pypi.org/simple/ml-collections/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/ml-collections/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))) - skipping
ERROR: Could not find a version that satisfies the requirement ml-collections (from versions: none)
ERROR: No matching distribution found for ml-collections
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ml-collections
wget --no-check-certificate https://huggingface.co/GYX98/HW_weight/resolve/main/hw_houqi3-class2-3e-2-9_10-acc_checkpoint.bin
import os
import json
from glob import glob

def generate_image_quality_dataset(root_dir, output_json="image_quality_dataset.json"):
    dataset = []
    
    # 遍历 True 和 False 子文件夹
    for label in ["True", "False"]:
        label_dir = os.path.join(root_dir, label)
        if not os.path.exists(label_dir):
            continue
        
        # 获取所有图像文件
        image_files = glob(os.path.join(label_dir, "*.[bB][mM][pP]"))  # 支持 .bmp 和 .BMP
        image_files.extend(glob(os.path.join(label_dir, "*.[jJ][pP][gG]")))  # 支持 .jpg 和 .JPG
        image_files.extend(glob(os.path.join(label_dir, "*.[jJ][pP][eE][gG]")))  # 支持 .jpeg 和 .JPEG
        image_files.extend(glob(os.path.join(label_dir, "*.[pP][nN][gG]")))  # 支持 .png 和 .PNG
        
        for image_path in image_files:
            # 获取相对路径
            rel_path = os.path.relpath(image_path, root_dir).replace("\\", "/")
            
            # 根据文件夹设置评价和分数
            if label == "True":
                quality_assessment = "The quality of the image is bad."
                gt_score = 3
            else:
                quality_assessment = "The quality of the image is excellent."
                gt_score = 5
            
            # 添加到数据集
            dataset.append({
                "id": f"{rel_path}->{gt_score}",
                "image": rel_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": "How would you rate the quality of this image?\n<|image|>"
                    },
                    {
                        "from": "gpt",
                        "value": quality_assessment
                    }
                ],
                "gt_score": gt_score
            })
    
    # 保存为 JSON 文件
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
    
    print(f"Dataset generated with {len(dataset)} entries. Saved to {output_json}")

# 使用示例
if __name__ == "__main__":
    # 替换为您的图像文件夹路径
    image_root_dir = "清晰度"  # 文件夹结构应为: 清晰度/True/... 和 清晰度/False/...
    generate_image_quality_dataset(image_root_dir)


  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'
Traceback (most recent call last):
  File "/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main/q_align/train/train_mem.py", line 34, in <module>
    from q_align.train.mplug_owl2_trainer import MPLUGOwl2Trainer
ModuleNotFoundError: No module named 'q_align'

export PYTHONPATH="/home/wsw/jikaiyuan/code/code_hw/realistic_postprocess/Q-Align-main:$PYTHONPATH"
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from PIL import Image
import torch

import pandas as pd
from pyiqa import create_metric
import cv2

def calculate_image_quality(model, image_np):
    """
    计算numpy格式图像的质量分数
    
    参数:
        weight_path: 模型权重路径
        image_np: numpy格式的图像数组 (H, W, C)或(N, H, W, C)
    """

    
    # 确保输入是4维 (N, H, W, C)
    if len(image_np.shape) == 3:
        image_np = np.expand_dims(image_np, axis=0)  # 添加batch维度
        
    # 收集结果
    results = []
    
    for i in range(image_np.shape[0]):
        # 转换为PIL Image
        img = Image.fromarray(image_np[i]).convert("RGB")
        
        # 计算质量分数
        with torch.no_grad():
            quality_score = model.score([img], task_="quality", input_="image")
            # aesthetics_score = model.score([img], task_="aesthetics", input_="image")

    return quality_score.item()
    
def calculate_image_quality_qualiclip(image_np):
    """Inference demo for pyiqa."""
    metric_name = 'qualiclip+'
    # 确保输入是4维 (N, H, W, C)
    if len(image_np.shape) == 3:
        image_np = np.expand_dims(image_np, axis=0)  # 添加batch维度
        
    for i in range(image_np.shape[0]):
        # 转换为PIL Image
        img = Image.fromarray(image_np[i]).convert("RGB")
        

    # set up IQA model
    iqa_model = create_metric(
        metric_name, metric_mode='NR', device=None
    )

    ref_img_path = None
    score = iqa_model(img, ref_img_path).cpu().item()
    return score
    
def is_bad_quality(img,model):
        #####清晰度，真实感，后期处理，需要转换图像
    # 1. 转换通道顺序: BGR -> RGB（如果模型需要RGB输入）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # shape: (H, W, C), RGB格式
    
    # 2. 添加batch维度 (N=1)
    img_np = np.expand_dims(img_rgb, axis=0)  # shape: (1, H, W, C)
    
    image_quality_qalign = calculate_image_quality(model,img_np)#需要从q-future/one-align下载相应权重
    image_quality_qualiclip = calculate_image_quality_qualiclip(img_np)
    
    # if (image_quality_qalign < 4.3) & (image_quality_qualiclip < 0.54):
    if (image_quality_qualiclip < 0.53)| (image_quality_qalign < 3.5):
        return True
    else:
        return False
        
# 使用示例
if __name__ == "__main__":
    # 加载图像为numpy数组的示例
    from PIL import Image
    img_path = "/mnt/sda/gyx/huawei_ad/3_6/y60073309_102_258208558_10_20250210192633.jpg"
    img_np = cv2.imread(img_path)  # (H, W, C)
    
    results = is_bad_quality(img_np)
    print(results)
    
# image_quality_qualiclip < 0.52
# === 规则 rule_5（清晰度差）统计 ===
# 总数: 100 | TP: 40 | FP: 1 | TN: 49 | FN: 10
# Accuracy: 0.8900 | Precision: 0.9756 | Recall: 0.8000 | F1: 0.8791

#  (image_quality_qualiclip < 0.53)| (image_quality_qalign < 3.5):
# === 规则 rule_5（清晰度差）统计 ===
# 总数: 100 | TP: 46 | FP: 4 | TN: 46 | FN: 4
# Accuracy: 0.9200 | Precision: 0.9200 | Recall: 0.9200 | F1: 0.9200


  [70/73] /usr/local/cuda/bin/nvcc  -I/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/csrc/flash_attn -I/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/csrc/flash_attn/src -I/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/csrc/cutlass/include -I/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/include -I/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/include/TH -I/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/include/THC -I/usr/local/cuda/include -I/root/miniconda3/envs/q-align/include/python3.10 -c -c /tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.cu -o /tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/build/temp.linux-x86_64-cpython-310/csrc/flash_attn/src/flash_fwd_split_hdim64_fp16_sm80.o -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr --compiler-options ''"'"'-fPIC'"'"'' -O3 -std=c++17 -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 --threads 4 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=flash_attn_2_cuda -D_GLIBCXX_USE_CXX11_ABI=0
      ninja: build stopped: subcommand failed.
      Traceback (most recent call last):
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 1348, in do_open
          h.request(req.get_method(), req.selector, req.data, headers,
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 1283, in request
          self._send_request(method, url, body, headers, encode_chunked)
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 1329, in _send_request
          self.endheaders(body, encode_chunked=encode_chunked)
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 1278, in endheaders
          self._send_output(message_body, encode_chunked=encode_chunked)
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 1038, in _send_output
          self.send(msg)
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 976, in send
          self.connect()
        File "/root/miniconda3/envs/q-align/lib/python3.10/http/client.py", line 1455, in connect
          self.sock = self._context.wrap_socket(self.sock,
        File "/root/miniconda3/envs/q-align/lib/python3.10/ssl.py", line 513, in wrap_socket
          return self.sslsocket_class._create(
        File "/root/miniconda3/envs/q-align/lib/python3.10/ssl.py", line 1104, in _create
          self.do_handshake()
        File "/root/miniconda3/envs/q-align/lib/python3.10/ssl.py", line 1375, in do_handshake
          self._sslobj.do_handshake()
      ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1017)
      
      During handling of the above exception, another exception occurred:
      
      Traceback (most recent call last):
        File "/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/setup.py", line 486, in run
          urllib.request.urlretrieve(wheel_url, wheel_filename)
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 241, in urlretrieve
          with contextlib.closing(urlopen(url, data)) as fp:
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 216, in urlopen
          return opener.open(url, data, timeout)
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 519, in open
          response = self._open(req, data)
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 536, in _open
          result = self._call_chain(self.handle_open, protocol, protocol +
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 496, in _call_chain
          result = func(*args)
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 1391, in https_open
          return self.do_open(http.client.HTTPSConnection, req,
        File "/root/miniconda3/envs/q-align/lib/python3.10/urllib/request.py", line 1351, in do_open
          raise URLError(err)
      urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: self-signed certificate in certificate chain (_ssl.c:1017)>
      
      During handling of the above exception, another exception occurred:
      
      Traceback (most recent call last):
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2100, in _run_ninja_build
          subprocess.run(
        File "/root/miniconda3/envs/q-align/lib/python3.10/subprocess.py", line 526, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['ninja', '-v', '-j', '70']' returned non-zero exit status 1.
      
      The above exception was the direct cause of the following exception:
      
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 35, in <module>
        File "/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/setup.py", line 526, in <module>
          setup(
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/__init__.py", line 117, in setup
          return distutils.core.setup(**attrs)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
          return run_commands(dist)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
          dist.run_commands()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
          self.run_command(cmd)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/dist.py", line 1104, in run_command
          super().run_command(command)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/tmp/pip-install-qhm5vj1y/flash-attn_98b1d7f138bc42bbb1d23fe62fcd949d/setup.py", line 503, in run
          super().run()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/command/bdist_wheel.py", line 370, in run
          self.run_command("build")
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/dist.py", line 1104, in run_command
          super().run_command(command)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/command/build.py", line 135, in run
          self.run_command(cmd_name)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 357, in run_command
          self.distribution.run_command(command)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/dist.py", line 1104, in run_command
          super().run_command(command)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
          cmd_obj.run()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 99, in run
          _build_ext.run(self)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 368, in run
          self.build_extensions()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 873, in build_extensions
          build_ext.build_extensions(self)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 484, in build_extensions
          self._build_extensions_serial()
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 510, in _build_extensions_serial
          self.build_extension(ext)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 264, in build_extension
          _build_ext.build_extension(self, ext)
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 565, in build_extension
          objects = self.compiler.compile(
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 686, in unix_wrap_ninja_compile
          _write_ninja_file_and_compile_objects(
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1774, in _write_ninja_file_and_compile_objects
          _run_ninja_build(
        File "/root/miniconda3/envs/q-align/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 2116, in _run_ninja_build
          raise RuntimeError(message) from e
      RuntimeError: Error compiling objects for extension
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for flash-attn
  Running setup.py clean for flash-attn
Failed to build flash-attn
error: failed-wheel-build-for-install

× Failed to build installable wheels for some pyproject.toml based projects
╰─> flash-attn
pip install flash-attn -i https://mirrors.aliyun.com/pypi/simple/

