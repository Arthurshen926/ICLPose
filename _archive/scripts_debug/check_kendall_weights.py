"""检查EXP009的checkpoint，查看Kendall Loss权重"""
import torch
import os

checkpoint_path = "output/exp009/checkpoints/latest.pth"

if os.path.exists(checkpoint_path):
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nEpoch: {checkpoint['epoch']}")
    print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # 查找Kendall Loss的log_var参数
    print("\n" + "=" * 80)
    print("模型参数中的Kendall Loss权重:")
    print("=" * 80)
    
    model_state = checkpoint['model_state_dict']
    
    kendall_params = {}
    for key, value in model_state.items():
        if 'log_var' in key or 'kendall' in key.lower():
            kendall_params[key] = value
            print(f"\n{key}:")
            print(f"  Shape: {value.shape}")
            print(f"  Value: {value}")
            
            # 计算实际权重
            if 'log_var' in key:
                weight = torch.exp(-value)
                print(f"  Weight (exp(-log_var)): {weight}")
    
    if not kendall_params:
        print("\n未找到Kendall Loss参数，检查pose_loss中的参数...")
        
        # 查找pose_regressor和pose_loss相关参数
        for key in model_state.keys():
            if 'pose' in key.lower():
                print(f"  {key}: {model_state[key].shape}")
    
    print("\n" + "=" * 80)
    print("分析:")
    print("=" * 80)
    
    if kendall_params:
        log_var_rot_key = [k for k in kendall_params if 'rotation' in k]
        log_var_trans_key = [k for k in kendall_params if 'translation' in k]
        
        if log_var_rot_key and log_var_trans_key:
            log_var_rot = kendall_params[log_var_rot_key[0]]
            log_var_trans = kendall_params[log_var_trans_key[0]]
            
            weight_rot = torch.exp(-log_var_rot).item()
            weight_trans = torch.exp(-log_var_trans).item()
            
            print(f"\n旋转权重: {weight_rot:.6f}")
            print(f"平移权重: {weight_trans:.6f}")
            print(f"权重比 (rot/trans): {weight_rot/weight_trans:.2f}x")
            
            if weight_rot > weight_trans * 2:
                print("\n⚠️ 旋转权重显著大于平移权重!")
                print("   → 模型更关注优化旋转，平移误差可能被忽略")
            elif weight_trans > weight_rot * 2:
                print("\n⚠️ 平移权重显著大于旋转权重!")
                print("   → 模型更关注优化平移，旋转误差可能被忽略")
            else:
                print("\n✓ 旋转和平移权重相对平衡")
    else:
        print("未使用Kendall Loss或参数未保存在checkpoint中")
        
else:
    print(f"❌ Checkpoint不存在: {checkpoint_path}")
