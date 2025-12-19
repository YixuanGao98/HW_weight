 ValueError: Free memory on device (28.56/44.53 GiB) on startup is less than desired GPU memory utilization (0.9, 40.07 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
[rank0]:[W1219 02:00:45.515146938 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())
Traceback (most recent call last):
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/runpy.py", line 198, in _run_module_as_main
    return _run_code(code, main_globals, None,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/runpy.py", line 88, in _run_code
    exec(code, run_globals)
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 71, in <module>
    cli.main()
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 508, in main
    run()
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 358, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 310, in run_path
    return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 127, in _run_module_code
    _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
  File "/root/.vscode-server/extensions/ms-python.debugpy-2025.18.0/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 118, in _run_code
    exec(code, run_globals)
  File "/home/wsw/gyx/code_12.17/rule_14_vllm.py", line 330, in <module>
    main()
  File "/home/wsw/gyx/code_12.17/rule_14_vllm.py", line 237, in main
    llm = LLM(
          ^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 297, in __init__
    self.llm_engine = LLMEngine.from_engine_args(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 177, in from_engine_args
    return cls(vllm_config=vllm_config,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py", line 114, in __init__
    self.engine_core = EngineCoreClient.make_client(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 80, in make_client
    return SyncMPClient(vllm_config, executor_class, log_stats)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 602, in __init__
    super().__init__(
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 448, in __init__
    with launch_core_engines(vllm_config, executor_class,
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/contextlib.py", line 144, in __exit__
    next(self.gen)
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 732, in launch_core_engines
    wait_for_engine_startup(
  File "/root/miniconda3/envs/easyr1_v2/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 785, in wait_for_engine_startup
    raise RuntimeError("Engine core initialization failed. "import torch
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
