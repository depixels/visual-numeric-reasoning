import torch
import os
import sys
# 1. 引入 safetensors 加载器 (关键修改)
from safetensors.torch import load_file
from train_stage1 import VisionEmbeddingModel

def check_saved_model(output_dir, model_name):
    print(f"Checking directory: {output_dir}")
    
    bin_path = os.path.join(output_dir, "pytorch_model.bin")
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    
    weights_path = None
    load_method = None

    # 2. 自动判断是 safetensors 还是 bin
    if os.path.exists(safetensors_path):
        weights_path = safetensors_path
        load_method = "safetensors"
        print(f"Found weights (Safetensors): {weights_path}")
    elif os.path.exists(bin_path):
        weights_path = bin_path
        load_method = "torch_load"
        print(f"Found weights (PyTorch Bin): {weights_path}")
    else:
        print("❌ CRITICAL: No model weights found!")
        return

    # 检查 Processor (可选)
    if os.path.exists(os.path.join(output_dir, "preprocessor_config.json")):
        print("✅ Processor config found.")
    else:
        print("⚠️ Warning: Preprocessor config not found (normal for checkpoints, critical for final model).")

    print("Attempting to load weights into model structure...")
    try:
        # 初始化空模型
        model = VisionEmbeddingModel(model_name)
        
        # 3. 根据格式正确加载权重
        if load_method == "safetensors":
            # 关键：使用 load_file 读取 .safetensors
            state_dict = load_file(weights_path)
        else:
            # 只有 .bin 文件才用 torch.load
            # weights_only=False 是为了兼容性，防止某些旧格式报错
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        
        # 处理可能的 DDP 前缀 (module.)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # 加载进模型
        msg = model.load_state_dict(new_state_dict, strict=True)
        print("Load result:", msg)
        print("✅ SUCCESS: Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ FAILED to load model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_saved_model(
        output_dir="/data/hyz/workspace/rege_bench/runs/stage1_qwen3vl_vit", 
        model_name="/data/hyz/workspace/hf/Qwen3-VL-4B-Instruct"
    )