# 通过 softmax 转化为概率分布
output_prob = softmax(output, dim=-1)

# 选择概率最高的类别作为预测类别
predicted_labels = torch.argmax(output_prob, dim=-1)

# 计算 Precision 和 Recall
precision = precision_score(target.numpy(), predicted_labels.numpy())
recall = recall_score(target.numpy(), predicted_labels.numpy())

def is_realistic_or_postprocessing(pretrained_dir,img_np):

    # 1. 转换通道顺序: BGR -> RGB（如果模型需要RGB输入）
    img_rgb = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))  # shape: (H, W, C), RGB格式
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # Normalize in pre-process of vqkd
    ])
    print(f"Image transforms: {transform}")

    images = transform(img_rgb).unsqueeze(0)

    # ============ building network ... ============
    model = create_model(
            'beit_base_patch16_224',
            pretrained=False,
#            as_tokenzer=True,
        ).eval()
    model.load_state_dict(torch.load(pretrained_dir))

    input_ids = model.get_codebook_indices(images)
    print(input_ids)




  File "/home/wsw/gyx/unilm-master/beit2/all.py", line 278, in main
    pred = bool(fn(img, running_idx))
                ^^^^^^^^^^^^^^^^^^^^
  File "/home/wsw/gyx/unilm-master/beit2/all.py", line 198, in <lambda>
    "fn": lambda img, idx=None: bool(is_realistic_or_postprocessing(POSTPRO_BEST,img))
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wsw/gyx/unilm-master/beit2/rule_6_7.py", line 44, in is_realistic_or_postprocessing
    model.load_state_dict(torch.load(pretrained_dir))
  File "/root/miniconda3/lib/python3.12/site-packages/torch/nn/modules/module.py", line 2584, in load_state_dict
    raise RuntimeError(
RuntimeError: Error(s) in loading state_dict for Beit:
        Missing key(s) in state_dict: "cls_token", "patch_embed.proj.weight", "patch_embed.proj.bias", "blocks.0.gamma_1", "blocks.0.gamma_2", "blocks.0.norm1.weight", "blocks.0.norm1.bias", "blocks.0.attn.q_bias", "blocks.0.attn.v_bias", "blocks.0.attn.relative_position_bias_table", "blocks.0.attn.qkv.weight", "blocks.0.attn.proj.weight", "blocks.0.attn.proj.bias", "blocks.0.norm2.weight", "blocks.0.norm2.bias", "blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc1.bias", "blocks.0.mlp.fc2.weight", "blocks.0.mlp.fc2.bias", "blocks.1.gamma_1", "blocks.1.gamma_2", "blocks.1.norm1.weight", "blocks.1.norm1.bias", "blocks.1.attn.q_bias", "blocks.1.attn.v_bias", "blocks.1.attn.relative_position_bias_table", "blocks.1.attn.qkv.weight", "blocks.1.attn.proj.weight", "blocks.1.attn.proj.bias", "blocks.1.norm2.weight", "blocks.1.norm2.bias", "blocks.1.mlp.fc1.weight", "blocks.1.mlp.fc1.bias", "blocks.1.mlp.fc2.weight", "blocks.1.mlp.fc2.bias", "blocks.2.gamma_1", "blocks.2.gamma_2", "blocks.2.norm1.weight", "blocks.2.norm1.bias", "blocks.2.attn.q_bias", "blocks.2.attn.v_bias", "blocks.2.attn.relative_position_bias_table", "blocks.2.attn.qkv.weight", "blocks.2.attn.proj.weight", "blocks.2.attn.proj.bias", "blocks.2.norm2.weight", "blocks.2.norm2.bias", "blocks.2.mlp.fc1.weight", "blocks.2.mlp.fc1.bias", "blocks.2.mlp.fc2.weight", "blocks.2.mlp.fc2.bias", "blocks.3.gamma_1", "blocks.3.gamma_2", "blocks.3.norm1.weight", "blocks.3.norm1.bias", "blocks.3.attn.q_bias", "blocks.3.attn.v_bias", "blocks.3.attn.relative_position_bias_table", "blocks.3.attn.qkv.weight", "blocks.3.attn.proj.weight", "blocks.3.attn.proj.bias", "blocks.3.norm2.weight", "blocks.3.norm2.bias", "blocks.3.mlp.fc1.weight", "blocks.3.mlp.fc1.bias", "blocks.3.mlp.fc2.weight", "blocks.3.mlp.fc2.bias", "blocks.4.gamma_1", "blocks.4.gamma_2", "blocks.4.norm1.weight", "blocks.4.norm1.bias", "blocks.4.attn.q_bias", "blocks.4.attn.v_bias", "blocks.4.attn.relative_position_bias_table", "blocks.4.attn.qkv.weight", "blocks.4.attn.proj.weight", "blocks.4.attn.proj.bias", "blocks.4.norm2.weight", "blocks.4.norm2.bias", "blocks.4.mlp.fc1.weight", "blocks.4.mlp.fc1.bias", "blocks.4.mlp.fc2.weight", "blocks.4.mlp.fc2.bias", "blocks.5.gamma_1", "blocks.5.gamma_2", "blocks.5.norm1.weight", "blocks.5.norm1.bias", "blocks.5.attn.q_bias", "blocks.5.attn.v_bias", "blocks.5.attn.relative_position_bias_table", "blocks.5.attn.qkv.weight", "blocks.5.attn.proj.weight", "blocks.5.attn.proj.bias", "blocks.5.norm2.weight", "blocks.5.norm2.bias", "blocks.5.mlp.fc1.weight", "blocks.5.mlp.fc1.bias", "blocks.5.mlp.fc2.weight", "blocks.5.mlp.fc2.bias", "blocks.6.gamma_1", "blocks.6.gamma_2", "blocks.6.norm1.weight", "blocks.6.norm1.bias", "blocks.6.attn.q_bias", "blocks.6.attn.v_bias", "blocks.6.attn.relative_position_bias_table", "blocks.6.attn.qkv.weight", "blocks.6.attn.proj.weight", "blocks.6.attn.proj.bias", "blocks.6.norm2.weight", "blocks.6.norm2.bias", "blocks.6.mlp.fc1.weight", "blocks.6.mlp.fc1.bias", "blocks.6.mlp.fc2.weight", "blocks.6.mlp.fc2.bias", "blocks.7.gamma_1", "blocks.7.gamma_2", "blocks.7.norm1.weight", "blocks.7.norm1.bias", "blocks.7.attn.q_bias", "blocks.7.attn.v_bias", "blocks.7.attn.relative_position_bias_table", "blocks.7.attn.qkv.weight", "blocks.7.attn.proj.weight", "blocks.7.attn.proj.bias", "blocks.7.norm2.weight", "blocks.7.norm2.bias", "blocks.7.mlp.fc1.weight", "blocks.7.mlp.fc1.bias", "blocks.7.mlp.fc2.weight", "blocks.7.mlp.fc2.bias", "blocks.8.gamma_1", "blocks.8.gamma_2", "blocks.8.norm1.weight", "blocks.8.norm1.bias", "blocks.8.attn.q_bias", "blocks.8.attn.v_bias", "blocks.8.attn.relative_position_bias_table", "blocks.8.attn.qkv.weight", "blocks.8.attn.proj.weight", "blocks.8.attn.proj.bias", "blocks.8.norm2.weight", "blocks.8.norm2.bias", "blocks.8.mlp.fc1.weight", "blocks.8.mlp.fc1.bias", "blocks.8.mlp.fc2.weight", "blocks.8.mlp.fc2.bias", "blocks.9.gamma_1", "blocks.9.gamma_2", "blocks.9.norm1.weight", "blocks.9.norm1.bias", "blocks.9.attn.q_bias", "blocks.9.attn.v_bias", "blocks.9.attn.relative_position_bias_table", "blocks.9.attn.qkv.weight", "blocks.9.attn.proj.weight", "blocks.9.attn.proj.bias", "blocks.9.norm2.weight", "blocks.9.norm2.bias", "blocks.9.mlp.fc1.weight", "blocks.9.mlp.fc1.bias", "blocks.9.mlp.fc2.weight", "blocks.9.mlp.fc2.bias", "blocks.10.gamma_1", "blocks.10.gamma_2", "blocks.10.norm1.weight", "blocks.10.norm1.bias", "blocks.10.attn.q_bias", "blocks.10.attn.v_bias", "blocks.10.attn.relative_position_bias_table", "blocks.10.attn.qkv.weight", "blocks.10.attn.proj.weight", "blocks.10.attn.proj.bias", "blocks.10.norm2.weight", "blocks.10.norm2.bias", "blocks.10.mlp.fc1.weight", "blocks.10.mlp.fc1.bias", "blocks.10.mlp.fc2.weight", "blocks.10.mlp.fc2.bias", "blocks.11.gamma_1", "blocks.11.gamma_2", "blocks.11.norm1.weight", "blocks.11.norm1.bias", "blocks.11.attn.q_bias", "blocks.11.attn.v_bias", "blocks.11.attn.relative_position_bias_table", "blocks.11.attn.qkv.weight", "blocks.11.attn.proj.weight", "blocks.11.attn.proj.bias", "blocks.11.norm2.weight", "blocks.11.norm2.bias", "blocks.11.mlp.fc1.weight", "blocks.11.mlp.fc1.bias", "blocks.11.mlp.fc2.weight", "blocks.11.mlp.fc2.bias", "fc_norm.weight", "fc_norm.bias". 
        Unexpected key(s) in state_dict: "transformer.embeddings.position_embeddings", "transformer.embeddings.cls_token", "transformer.embeddings.hybrid_model.root.conv.weight", "transformer.embeddings.hybrid_model.root.gn.weight", "transformer.embeddings.hybrid_model.root.gn.bias", "transformer.embeddings.hybrid_model.body.block1.unit1.gn1.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn1.bias", "transformer.embeddings.hybrid_model.body.block1.unit1.conv1.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn2.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn2.bias", "transformer.embeddings.hybrid_model.body.block1.unit1.conv2.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn3.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn3.bias", "transformer.embeddings.hybrid_model.body.block1.unit1.conv3.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.downsample.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn_proj.weight", "transformer.embeddings.hybrid_model.body.block1.unit1.gn_proj.bias", "transformer.embeddings.hybrid_model.body.block1.unit2.gn1.weight", "transformer.embeddings.hybrid_model.body.block1.unit2.gn1.bias", "transformer.embeddings.hybrid_model.body.block1.unit2.conv1.weight", "transformer.embeddings.hybrid_model.body.block1.unit2.gn2.weight", "transformer.embeddings.hybrid_model.body.block1.unit2.gn2.bias", "transformer.embeddings.hybrid_model.body.block1.unit2.conv2.weight", "transformer.embeddings.hybrid_model.body.block1.unit2.gn3.weight", "transformer.embeddings.hybrid_model.body.block1.unit2.gn3.bias", "transformer.embeddings.hybrid_model.body.block1.unit2.conv3.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn1.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn1.bias", "transformer.embeddings.hybrid_model.body.block1.unit3.conv1.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn2.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn2.bias", "transformer.embeddings.hybrid_model.body.block1.unit3.conv2.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn3.weight", "transformer.embeddings.hybrid_model.body.block1.unit3.gn3.bias", "transformer.embeddings.hybrid_model.body.block1.unit3.conv3.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn1.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn1.bias", "transformer.embeddings.hybrid_model.body.block2.unit1.conv1.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn2.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn2.bias", "transformer.embeddings.hybrid_model.body.block2.unit1.conv2.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn3.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn3.bias", "transformer.embeddings.hybrid_model.body.block2.unit1.conv3.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.downsample.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn_proj.weight", "transformer.embeddings.hybrid_model.body.block2.unit1.gn_proj.bias", "transformer.embeddings.hybrid_model.body.block2.unit2.gn1.weight", "transformer.embeddings.hybrid_model.body.block2.unit2.gn1.bias", "transformer.embeddings.hybrid_model.body.block2.unit2.conv1.weight", "transformer.embeddings.hybrid_model.body.block2.unit2.gn2.weight", "transformer.embeddings.hybrid_model.body.block2.unit2.gn2.bias", "transformer.embeddings.hybrid_model.body.block2.unit2.conv2.weight", "transformer.embeddings.hybrid_model.body.block2.unit2.gn3.weight", "transformer.embeddings.hybrid_model.body.block2.unit2.gn3.bias", "transformer.embeddings.hybrid_model.body.block2.unit2.conv3.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn1.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn1.bias", "transformer.embeddings.hybrid_model.body.block2.unit3.conv1.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn2.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn2.bias", "transformer.embeddings.hybrid_model.body.block2.unit3.conv2.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn3.weight", "transformer.embeddings.hybrid_model.body.block2.unit3.gn3.bias", "transformer.embeddings.hybrid_model.body.block2.unit3.conv3.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn1.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn1.bias", "transformer.embeddings.hybrid_model.body.block2.unit4.conv1.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn2.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn2.bias", "transformer.embeddings.hybrid_model.body.block2.unit4.conv2.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn3.weight", "transformer.embeddings.hybrid_model.body.block2.unit4.gn3.bias", "transformer.embeddings.hybrid_model.body.block2.unit4.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit1.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit1.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit1.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.downsample.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn_proj.weight", "transformer.embeddings.hybrid_model.body.block3.unit1.gn_proj.bias", "transformer.embeddings.hybrid_model.body.block3.unit2.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit2.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit2.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit2.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit2.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit2.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit2.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit2.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit2.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit3.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit3.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit3.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit3.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit4.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit4.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit4.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit4.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit5.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit5.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit5.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit5.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit6.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit6.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit6.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit6.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit7.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit7.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit7.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit7.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit8.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit8.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit8.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit8.conv3.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn1.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn1.bias", "transformer.embeddings.hybrid_model.body.block3.unit9.conv1.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn2.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn2.bias", "transformer.embeddings.hybrid_model.body.block3.unit9.conv2.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn3.weight", "transformer.embeddings.hybrid_model.body.block3.unit9.gn3.bias", "transformer.embeddings.hybrid_model.body.block3.unit9.conv3.weight", "transformer.embeddings.patch_embeddings.weight", "transformer.embeddings.patch_embeddings.bias", "transformer.encoder.layer.0.attention_norm.weight", "transformer.encoder.layer.0.attention_norm.bias", "transformer.encoder.layer.0.ffn_norm.weight", "transformer.encoder.layer.0.ffn_norm.bias", "transformer.encoder.layer.0.ffn.fc1.weight", "transformer.encoder.layer.0.ffn.fc1.bias", "transformer.encoder.layer.0.ffn.fc2.weight", "transformer.encoder.layer.0.ffn.fc2.bias", "transformer.encoder.layer.0.attn.query.weight", "transformer.encoder.layer.0.attn.query.bias", "transformer.encoder.layer.0.attn.key.weight", "transformer.encoder.layer.0.attn.key.bias", "transformer.encoder.layer.0.attn.value.weight", "transformer.encoder.layer.0.attn.value.bias", "transformer.encoder.layer.0.attn.out.weight", "transformer.encoder.layer.0.attn.out.bias", "transformer.encoder.layer.1.attention_norm.weight", "transformer.encoder.layer.1.attention_norm.bias", "transformer.encoder.layer.1.ffn_norm.weight", "transformer.encoder.layer.1.ffn_norm.bias", "transformer.encoder.layer.1.ffn.fc1.weight", "transformer.encoder.layer.1.ffn.fc1.bias", "transformer.encoder.layer.1.ffn.fc2.weight", "transformer.encoder.layer.1.ffn.fc2.bias", "transformer.encoder.layer.1.attn.query.weight", "transformer.encoder.layer.1.attn.query.bias", "transformer.encoder.layer.1.attn.key.weight", "transformer.encoder.layer.1.attn.key.bias", "transformer.encoder.layer.1.attn.value.weight", "transformer.encoder.layer.1.attn.value.bias", "transformer.encoder.layer.1.attn.out.weight", "transformer.encoder.layer.1.attn.out.bias", "transformer.encoder.layer.2.attention_norm.weight", "transformer.encoder.layer.2.attention_norm.bias", "transformer.encoder.layer.2.ffn_norm.weight", "transformer.encoder.layer.2.ffn_norm.bias", "transformer.encoder.layer.2.ffn.fc1.weight", "transformer.encoder.layer.2.ffn.fc1.bias", "transformer.encoder.layer.2.ffn.fc2.weight", "transformer.encoder.layer.2.ffn.fc2.bias", "transformer.encoder.layer.2.attn.query.weight", "transformer.encoder.layer.2.attn.query.bias", "transformer.encoder.layer.2.attn.key.weight", "transformer.encoder.layer.2.attn.key.bias", "transformer.encoder.layer.2.attn.value.weight", "transformer.encoder.layer.2.attn.value.bias", "transformer.encoder.layer.2.attn.out.weight", "transformer.encoder.layer.2.attn.out.bias", "transformer.encoder.layer.3.attention_norm.weight", "transformer.encoder.layer.3.attention_norm.bias", "transformer.encoder.layer.3.ffn_norm.weight", "transformer.encoder.layer.3.ffn_norm.bias", "transformer.encoder.layer.3.ffn.fc1.weight", "transformer.encoder.layer.3.ffn.fc1.bias", "transformer.encoder.layer.3.ffn.fc2.weight", "transformer.encoder.layer.3.ffn.fc2.bias", "transformer.encoder.layer.3.attn.query.weight", "transformer.encoder.layer.3.attn.query.bias", "transformer.encoder.layer.3.attn.key.weight", "transformer.encoder.layer.3.attn.key.bias", "transformer.encoder.layer.3.attn.value.weight", "transformer.encoder.layer.3.attn.value.bias", "transformer.encoder.layer.3.attn.out.weight", "transformer.encoder.layer.3.attn.out.bias", "transformer.encoder.layer.4.attention_norm.weight", "transformer.encoder.layer.4.attention_norm.bias", "transformer.encoder.layer.4.ffn_norm.weight", "transformer.encoder.layer.4.ffn_norm.bias", "transformer.encoder.layer.4.ffn.fc1.weight", "transformer.encoder.layer.4.ffn.fc1.bias", "transformer.encoder.layer.4.ffn.fc2.weight", "transformer.encoder.layer.4.ffn.fc2.bias", "transformer.encoder.layer.4.attn.query.weight", "transformer.encoder.layer.4.attn.query.bias", "transformer.encoder.layer.4.attn.key.weight", "transformer.encoder.layer.4.attn.key.bias", "transformer.encoder.layer.4.attn.value.weight", "transformer.encoder.layer.4.attn.value.bias", "transformer.encoder.layer.4.attn.out.weight", "transformer.encoder.layer.4.attn.out.bias", "transformer.encoder.layer.5.attention_norm.weight", "transformer.encoder.layer.5.attention_norm.bias", "transformer.encoder.layer.5.ffn_norm.weight", "transformer.encoder.layer.5.ffn_norm.bias", "transformer.encoder.layer.5.ffn.fc1.weight", "transformer.encoder.layer.5.ffn.fc1.bias", "transformer.encoder.layer.5.ffn.fc2.weight", "transformer.encoder.layer.5.ffn.fc2.bias", "transformer.encoder.layer.5.attn.query.weight", "transformer.encoder.layer.5.attn.query.bias", "transformer.encoder.layer.5.attn.key.weight", "transformer.encoder.layer.5.attn.key.bias", "transformer.encoder.layer.5.attn.value.weight", "transformer.encoder.layer.5.attn.value.bias", "transformer.encoder.layer.5.attn.out.weight", "transformer.encoder.layer.5.attn.out.bias", "transformer.encoder.layer.6.attention_norm.weight", "transformer.encoder.layer.6.attention_norm.bias", "transformer.encoder.layer.6.ffn_norm.weight", "transformer.encoder.layer.6.ffn_norm.bias", "transformer.encoder.layer.6.ffn.fc1.weight", "transformer.encoder.layer.6.ffn.fc1.bias", "transformer.encoder.layer.6.ffn.fc2.weight", "transformer.encoder.layer.6.ffn.fc2.bias", "transformer.encoder.layer.6.attn.query.weight", "transformer.encoder.layer.6.attn.query.bias", "transformer.encoder.layer.6.attn.key.weight", "transformer.encoder.layer.6.attn.key.bias", "transformer.encoder.layer.6.attn.value.weight", "transformer.encoder.layer.6.attn.value.bias", "transformer.encoder.layer.6.attn.out.weight", "transformer.encoder.layer.6.attn.out.bias", "transformer.encoder.layer.7.attention_norm.weight", "transformer.encoder.layer.7.attention_norm.bias", "transformer.encoder.layer.7.ffn_norm.weight", "transformer.encoder.layer.7.ffn_norm.bias", "transformer.encoder.layer.7.ffn.fc1.weight", "transformer.encoder.layer.7.ffn.fc1.bias", "transformer.encoder.layer.7.ffn.fc2.weight", "transformer.encoder.layer.7.ffn.fc2.bias", "transformer.encoder.layer.7.attn.query.weight", "transformer.encoder.layer.7.attn.query.bias", "transformer.encoder.layer.7.attn.key.weight", "transformer.encoder.layer.7.attn.key.bias", "transformer.encoder.layer.7.attn.value.weight", "transformer.encoder.layer.7.attn.value.bias", "transformer.encoder.layer.7.attn.out.weight", "transformer.encoder.layer.7.attn.out.bias", "transformer.encoder.layer.8.attention_norm.weight", "transformer.encoder.layer.8.attention_norm.bias", "transformer.encoder.layer.8.ffn_norm.weight", "transformer.encoder.layer.8.ffn_norm.bias", "transformer.encoder.layer.8.ffn.fc1.weight", "transformer.encoder.layer.8.ffn.fc1.bias", "transformer.encoder.layer.8.ffn.fc2.weight", "transformer.encoder.layer.8.ffn.fc2.bias", "transformer.encoder.layer.8.attn.query.weight", "transformer.encoder.layer.8.attn.query.bias", "transformer.encoder.layer.8.attn.key.weight", "transformer.encoder.layer.8.attn.key.bias", "transformer.encoder.layer.8.attn.value.weight", "transformer.encoder.layer.8.attn.value.bias", "transformer.encoder.layer.8.attn.out.weight", "transformer.encoder.layer.8.attn.out.bias", "transformer.encoder.layer.9.attention_norm.weight", "transformer.encoder.layer.9.attention_norm.bias", "transformer.encoder.layer.9.ffn_norm.weight", "transformer.encoder.layer.9.ffn_norm.bias", "transformer.encoder.layer.9.ffn.fc1.weight", "transformer.encoder.layer.9.ffn.fc1.bias", "transformer.encoder.layer.9.ffn.fc2.weight", "transformer.encoder.layer.9.ffn.fc2.bias", "transformer.encoder.layer.9.attn.query.weight", "transformer.encoder.layer.9.attn.query.bias", "transformer.encoder.layer.9.attn.key.weight", "transformer.encoder.layer.9.attn.key.bias", "transformer.encoder.layer.9.attn.value.weight", "transformer.encoder.layer.9.attn.value.bias", "transformer.encoder.layer.9.attn.out.weight", "transformer.encoder.layer.9.attn.out.bias", "transformer.encoder.layer.10.attention_norm.weight", "transformer.encoder.layer.10.attention_norm.bias", "transformer.encoder.layer.10.ffn_norm.weight", "transformer.encoder.layer.10.ffn_norm.bias", "transformer.encoder.layer.10.ffn.fc1.weight", "transformer.encoder.layer.10.ffn.fc1.bias", "transformer.encoder.layer.10.ffn.fc2.weight", "transformer.encoder.layer.10.ffn.fc2.bias", "transformer.encoder.layer.10.attn.query.weight", "transformer.encoder.layer.10.attn.query.bias", "transformer.encoder.layer.10.attn.key.weight", "transformer.encoder.layer.10.attn.key.bias", "transformer.encoder.layer.10.attn.value.weight", "transformer.encoder.layer.10.attn.value.bias", "transformer.encoder.layer.10.attn.out.weight", "transformer.encoder.layer.10.attn.out.bias", "transformer.encoder.layer.11.attention_norm.weight", "transformer.encoder.layer.11.attention_norm.bias", "transformer.encoder.layer.11.ffn_norm.weight", "transformer.encoder.layer.11.ffn_norm.bias", "transformer.encoder.layer.11.ffn.fc1.weight", "transformer.encoder.layer.11.ffn.fc1.bias", "transformer.encoder.layer.11.ffn.fc2.weight", "transformer.encoder.layer.11.ffn.fc2.bias", "transformer.encoder.layer.11.attn.query.weight", "transformer.encoder.layer.11.attn.query.bias", "transformer.encoder.layer.11.attn.key.weight", "transformer.encoder.layer.11.attn.key.bias", "transformer.encoder.layer.11.attn.value.weight", "transformer.encoder.layer.11.attn.value.bias", "transformer.encoder.layer.11.attn.out.weight", "transformer.encoder.layer.11.attn.out.bias", "transformer.encoder.encoder_norm.weight", "transformer.encoder.encoder_norm.bias". 
        size mismatch for head.weight: copying a param with shape torch.Size([100, 768]) from checkpoint, the shape in current model is torch.Size([1000, 768]).
        size mismatch for head.bias: copying a param with shape torch.Size([100]) from checkpoint, the shape in current model is torch.Size([1000]).


import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(1024, 768)):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入文件夹
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            img_path = os.path.join(input_dir, filename)
            out_path = os.path.join(output_dir, filename)

            try:
                # 打开并resize
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize(size, Image.LANCZOS)  # 高质量resize

                # 保存到输出文件夹
                img_resized.save(out_path)
                print(f"✅ {filename} 已保存到 {out_path}")
            except Exception as e:
                print(f"❌ 处理 {filename} 出错: {e}")

# 使用方法
input_folder = "input_images"     # 你的输入文件夹路径
output_folder = "resized_images"  # 输出文件夹路径
resize_images(input_folder, output_folder)

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
https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets/resolve/main/koniq10k.tgz
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
pip install flash-attn --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org
3.970703125
0.47054266929626465
[rule_5/True] 202508011724294810999433D646578ED24A67EE1B58F3.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.08984375
0.5446516275405884
[rule_5/True] 202508011724300D54C161E4504FA0B0044A8D1CD8F13B.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.12890625
0.5347238779067993
[rule_5/True] 202508011744028E30F6A2B2A048EC8EB40E051ABA29F8.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.9921875
0.4027281403541565
[rule_5/True] 20250801175350C0736DA86A5A4AD09E4E8700948B6BE6.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.2578125
0.4544339179992676
[rule_5/True] 2025082218001773e6cf299c294f8aa622e42cb5384eb9.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.15625
0.6147210597991943
[rule_5/True] 20250822194641F15F58FBBA07452DA821C9F22C23DF13.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.46484375
0.42878609895706177
[rule_5/True] 20250902183332865BF2860813448D987738FBB6C31082.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.359375
0.569950520992279
[rule_5/True] 2025090218494503A8B0D44DFA40E59E21661A43EDA802.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.953125
0.5036550164222717
[rule_5/True] 20250902184945A8AB36D9D4B84C64A334C324982B63C7.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.302734375
0.5524501800537109
[rule_5/True] 2025090515442942EE6C28615A4885835F69CABB3B4896.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.87109375
0.5707246661186218
[rule_5/True] 2025092318440421D6F5C8058E42A6B12EC81F91DB43DF.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3203125
0.5375510454177856
[rule_5/True] dsp_url_202508081750470EA553EE84CE4204AC7BA16EED243802.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
2.900390625
0.39633044600486755
[rule_5/True] dsp_url_202509231800066EBB8604E92D427996C59A51D047D660.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.9296875
0.6200064420700073
[rule_5/True] 文字促销感重_后期处理_精美度__202508081032427e60e7e5c5524cc89db6ce0aa29db9a3.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.1640625
0.6414284706115723
[rule_5/True] 文字促销感重_后期处理_精美度_清晰度__20250810103745a5cdbbd6c39d4d589f85e4f8338c3cd9.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.09765625
0.5880330204963684
[rule_5/True] 文字促销感重_后期处理_精美度_清晰度__202508101039468ecdcf5d7cf74d9e825dcb000f752f46.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.0625
0.5994113683700562
[rule_5/True] 文字促销感重_后期处理_精美度_清晰度__20250810104248cfda344ae0964edd90220b85f54771b0.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3203125
0.6053613424301147
[rule_5/True] 文字促销感重_后期处理_精美度_清晰度__20250810105444dd8414ace5c74b9d8bca3006ef055068.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.869140625
0.6210309267044067
[rule_5/True] 文字促销感重_清晰度_精美度__202508081025121c000e96521847d4b5efe8ca121841e4.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.466796875
0.5550448894500732
[rule_5/True] 文字促销感重_清晰度_精美度__20250808102750c2a20d98ae164312ae7396a8cd1c0b08.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.06640625
0.45870792865753174
[rule_5/True] 文字促销感重_清晰度_精美度__20250808104722d87083adcf4a4cf3843487466f9a280b.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.84375
0.5796284079551697
[rule_5/True] 文字促销感重_清晰度_精美度__20250808105203f7afd5a5760c4d3aa691ea7e1490cbb1.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.947265625
0.539668083190918
[rule_5/True] 文字促销感重_清晰度_精美度__202508101024561ae8647952c147b5a98c3232b9227784.jpg -> 预测:True | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.15625
0.6754626035690308
[rule_5/True] 文字促销感重_清晰度_精美度__20250810102714f9d641c402eb4c24851c868f59017e14.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.11328125
0.6291708946228027
[rule_5/True] 文字促销感重_清晰度_精美度__20250810103749c528d1aefb8449468a403f38d05e8d2b.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.953125
0.5675967931747437
[rule_5/True] 文字促销感重_清晰度_精美度__202508101040192204b8c441e24faeae247e2e25a2d69b.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.578125
0.5618650913238525
[rule_5/True] 文字促销感重_精美度_清晰度__20250808102753e29b23da6a56409392cead9b466965da.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.810546875
0.6455385684967041
[rule_5/True] 文字促销感重_精美度_清晰度（模棱两可额）__20250810102459b0237f1a0bbd4b1fa64734925bed0c8e.jpg -> 预测:False | 真实:True
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.36328125
0.5296385288238525
[rule_5/False] 202507171759420E3753EA5DE54D10A921F50543CD40B5.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.89453125
0.5457233786582947
[rule_5/False] 202507301128478BC608779F354EBA974D3A8BD9D886D7.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.39453125
0.6105613708496094
[rule_5/False] 2025080515171275A120FF9906452980D21795B82E1DA9.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.939453125
0.6072190403938293
[rule_5/False] 2025080609314806304AD5E6784CA1B9E5EEE1653F9AAB.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.31640625
0.6754438281059265
[rule_5/False] 20250813160846196787B34459415B9E089FB6C01B9090.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.1328125
0.6695600748062134
[rule_5/False] 2025081518124049b65db3ad2b42888f6061e8749e63a2.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.31640625
0.6534541845321655
[rule_5/False] 2025081810463743ed3efc3af141188679e417662a9a87.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.31640625
0.6560771465301514
[rule_5/False] 2025081811550156EEA4F44FED43639E64BF6B8A4A2B14.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.30859375
0.5469538569450378
[rule_5/False] 2025081818240278AE902D4EE64CDE83C1C885FD8AD815.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.2265625
0.676246702671051
[rule_5/False] 2025081821460640DAAC12103447E98D72630B09582F53.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.13671875
0.5913965702056885
[rule_5/False] 202508191452286873E4A7171F4781A19C1E3FDBB1F7A8.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.47265625
0.6039924025535583
[rule_5/False] 20250820112913CC4238BC69FA46CCB1CA8D29324CEA89.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.40234375
0.5815380811691284
[rule_5/False] 20250820112913E9C27F4B1CB8491FBAA72D9B5C4E7644.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.5078125
0.6584481000900269
[rule_5/False] 20250820112914F874B53A86C04723AADAD4508E3BAA08.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.58203125
0.5497040152549744
[rule_5/False] 20250820112914FCA8F17065FE4F5A8BE0B084150C2BDE.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.869140625
0.6247681975364685
[rule_5/False] 20250820174958C3DF67ACEE2A40E8A1AC0E2C19250DC6.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.31640625
0.6160472631454468
[rule_5/False] 20250820211118e029a3a354a8412587914f47f9679fb8.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.4765625
0.6383341550827026
[rule_5/False] 20250820211130f433ce25bbd14704884a27fe5590f220.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.46484375
0.6568992733955383
[rule_5/False] 20250820211209df0ee33cdeb04fa5aa8ffeb53f5988aa.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.34765625
0.5948393940925598
[rule_5/False] 20250820211212ff41d43d5cb74756923cc9bce6783c42.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.11328125
0.6143455505371094
[rule_5/False] 20250820211430880fbae6a16e4e1c9fcc84278ec85d87.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.27734375
0.6411608457565308
[rule_5/False] 20250820211539c1975a2b4929442385ef48a25bfc91b3.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.9765625
0.5430749654769897
[rule_5/False] 20250821145317E4398E4AED7F4F9F84CECA3EBEC7A025.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.0234375
0.5592329502105713
[rule_5/False] 20250821162700A87CA78F70204081B78A97CAA4B90176.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.23046875
0.5976365208625793
[rule_5/False] 20250821162701F3E96487C4094701902E230F1AB9DF64.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3984375
0.642775297164917
[rule_5/False] _20250818102728e0ba23aa8458464ab443e37a611ad4c2.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.578125
0.5777890682220459
[rule_5/False] _20250820112914F4C3AB9B5AE24BA7AB825E415CCCC054.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.08984375
0.5947120189666748
[rule_5/False] _202508201510250F0DEC26EEB9410B9D5901F1643273B2.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.12109375
0.6103329062461853
[rule_5/False] _20250820151025ED806B8BFD464AB6BAA257D3026342A6.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.39453125
0.6324851512908936
[rule_5/False] _20250820191913e89faac397544adfb2d4197905b6b501.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.44921875
0.6358564496040344
[rule_5/False] _202508201919171dc6b35eb4fb405bbdc082840647b5f8.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.58203125
0.6203743815422058
[rule_5/False] _20250820191918cd52197beccf463c930575eb07b25fe5.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.4453125
0.5525977611541748
[rule_5/False] _20250820191926eaf6ce70f7e940f983711037b0365ed4.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.66015625
0.6376363039016724
[rule_5/False] _20250820191931fd0f90922e9b46889192a58927a1c084.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.52734375
0.6232709884643555
[rule_5/False] _20250820191946eb706912b0164e88a564bd2933ac68a9.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.19140625
0.6935747861862183
[rule_5/False] _20250821162701AD57134C4FD247E0A8C9D66245D5656C.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.43359375
0.6418907046318054
[rule_5/False] _dsp_url_2025051917345945FB72FC174F4DE68B518B03B0E2B30D.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.0703125
0.6893547773361206
[rule_5/False] _dsp_url_2025081317582472794249E5EB429A89E439E81C8ED6A7.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.55078125
0.5912187099456787
[rule_5/False] _dsp_url_20250818150052614815232031446DB1C144BEBD86A58F.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.58984375
0.6013585925102234
[rule_5/False] _dsp_url_2025081914120668CC8FE7DABF46F68F564C6D1D0E4898.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.40625
0.6187252402305603
[rule_5/False] _dsp_url_2025081917073747973E7496294F9EAF9D919E5B8D1F65.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.73046875
0.682177722454071
[rule_5/False] _dsp_url_2025081917402282661CCDD99843DC86295824959A47DC.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3203125
0.5945342183113098
[rule_5/False] _dsp_url_2025082017431550D405823D514BBEADF224A1A62CC449.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.94140625
0.6286210417747498
[rule_5/False] _dsp_url_202508211444339367C6BC9E9141FD88E963CF6C002523.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.89453125
0.7123289704322815
[rule_5/False] _dsp_url_202508211702152FA4364E455546E2A6DA8BEC633847AD.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.65234375
0.5394849181175232
[rule_5/False] dsp_url_2025050910345170230B855F5E4104BAC156A81E7C4FD1.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.23828125
0.6365645527839661
[rule_5/False] dsp_url_2025051313434873998272D1374F9FA1480611BA672C4B.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.20703125
0.5906908512115479
[rule_5/False] dsp_url_20250625150356A2B11CA245CC4ACFB519ADF8442392D9.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.87109375
0.551888644695282
[rule_5/False] dsp_url_20250625150356A8F2EC02144F4EEB80B7C87C9CB454A3.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.19921875
0.6156619787216187
[rule_5/False] dsp_url_20250703160047BF29E5123CCD491F95BDCCBFF8682D23.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
3.990234375
0.5200100541114807
[rule_5/False] dsp_url_20250707162036DB9103539EF34CC291E760DC73ED802F.jpg -> 预测:True | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.09375
0.60791015625
[rule_5/False] dsp_url_20250715103328DECB09FE1E47405198672CAF9C4F4B21.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.70703125
0.6955819725990295
[rule_5/False] dsp_url_20250725112342AE576026F07C43FC9566B1C64882BF73.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.04296875
0.591606616973877
[rule_5/False] dsp_url_202507251124082169AE0D9CE2438B8488F82A8D2FE7F9.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.4296875
0.6082338094711304
[rule_5/False] dsp_url_20250728142250F6A6B214C924419E99FA783A7CCA4769.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.375
0.6657873392105103
[rule_5/False] dsp_url_202508081628154826AE2893F94599A57E008563B53310.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3828125
0.6404680013656616
[rule_5/False] dsp_url_202508141014528278E712B86E4FFC955EB64852F0CEB6.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.4609375
0.5639368295669556
[rule_5/False] dsp_url_20250815105251C2C57BA1FFFC453C908827C82993D836.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.34375
0.5710031390190125
[rule_5/False] dsp_url_20250815110819A964E29A88374B819F98CE2AB8B1CE4B.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3046875
0.5274462103843689
[rule_5/False] dsp_url_20250815110819B7DB619F7C59429ABCF99F5DDFCCD01A.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.58984375
0.5738198161125183
[rule_5/False] dsp_url_2025081511090977BACB55DB734F1EBD9D271FD20D246B.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.7421875
0.6277196407318115
[rule_5/False] dsp_url_2025081511385054A314F2CAB544FFA276B0975517A5F0.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.234375
0.6079986095428467
[rule_5/False] dsp_url_2025081518245011EF15EF38A349ADA8C7C86FE850747A.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.2421875
0.6181042790412903
[rule_5/False] dsp_url_2025081518275697CA0D43D6AC45F78B31C1128648D5EF.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.09375
0.6519144773483276
[rule_5/False] dsp_url_2025081518283250EC2BE34B354F30A30BC56EF690DF06.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.36328125
0.6428147554397583
[rule_5/False] dsp_url_2025081520011978B50753DEAC437FACF2B1DDCAA09A70.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.3046875
0.6866991519927979
[rule_5/False] dsp_url_20250815201011A96F6D34B9044DC6982511BF3B2A0275.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.13671875
0.5838228464126587
[rule_5/False] dsp_url_20250816204149CF2430E4D8B541DD84DF64BA8BEB857A.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.30078125
0.7102533578872681
[rule_5/False] dsp_url_20250817213054B518EFDED8634D6899F75990CD76C635.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.44921875
0.6474472284317017
[rule_5/False] dsp_url_20250818150308CA6A90E3A6BD422D83F4EB3BC8A5EC79.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.55859375
0.5755252242088318
[rule_5/False] dsp_url_20250818150308FA84BB3E955B451CAC4231DB9582E343.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.39453125
0.6203721165657043
[rule_5/False] dsp_url_20250819140833284B5521FA3143868FE411B113CA155A.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.34375
0.6238701939582825
[rule_5/False] dsp_url_20250820175051886C5855F6134FD994DB2E2D80442332.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.28515625
0.5302484035491943
[rule_5/False] dsp_url_20250820175745681A0D0274D94993854F4777F2502FAA.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.79296875
0.659087598323822
[rule_5/False] dsp_url_2025082114341313E50529AF1C4538B65AC09DE70E7F95.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.45703125
0.6934659481048584
[rule_5/False] dsp_url_2025082114360841B2191E7D1E4F089EBEB623024EC7AE.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.78125
0.7774208784103394
[rule_5/False] dsp_url_2025082116555265CE45116D6D4CFEBB84379BDFF6E765.jpg -> 预测:False | 真实:False
Loading pretrained model QualiCLIP from /root/.cache/torch/hub/pyiqa/QualiCLIP%2B_koniq.pth
4.69140625
0.6179515719413757
[rule_5/False] dsp_url_2025082117233880A83BB886164FA4AB64A2F0B01F47E6.jpg -> 预测:False | 真实:False
