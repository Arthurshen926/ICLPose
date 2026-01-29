#!/usr/bin/env python3
"""
训练环境检查脚本
在开始训练前运行此脚本，确保所有依赖和配置正确
"""

import os
import sys
from pathlib import Path
import yaml

def check_python_version():
    """检查Python版本"""
    print("检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ✗ Python版本过低: {version.major}.{version.minor}.{version.micro}")
        print(f"    需要Python 3.8+")
        return False

def check_pytorch():
    """检查PyTorch"""
    print("\n检查PyTorch...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA版本: {torch.version.cuda}")
            print(f"    可用GPU数量: {torch.cuda.device_count()}")
        else:
            print(f"  ⚠ CUDA不可用，将使用CPU训练（速度较慢）")
        
        return True
    except ImportError:
        print(f"  ✗ PyTorch未安装")
        print(f"    请运行: pip install torch torchvision")
        return False

def check_dependencies():
    """检查其他依赖"""
    print("\n检查其他依赖...")
    dependencies = [
        ('numpy', 'numpy'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('tensorboard', 'tensorboard'),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name}未安装")
            print(f"    请运行: pip install {package_name}")
            all_ok = False
    
    return all_ok

def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")
    required_dirs = [
        'data',
        'losses',
        'ic_models',
        'modules',
        'configs',
    ]
    
    required_files = [
        'train.py',
        'data/dataset.py',
        'losses/pose_loss.py',
        'ic_models/ic_pose_net.py',
        'configs/train_config.yaml',
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        if os.path.isdir(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ 不存在")
            all_ok = False
    
    for file_name in required_files:
        if os.path.isfile(file_name):
            print(f"  ✓ {file_name}")
        else:
            print(f"  ✗ {file_name} 不存在")
            all_ok = False
    
    return all_ok

def check_config_file(config_path='configs/train_config.yaml'):
    """检查配置文件"""
    print(f"\n检查配置文件: {config_path}...")
    
    if not os.path.exists(config_path):
        print(f"  ✗ 配置文件不存在: {config_path}")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ 配置文件格式正确")
        
        # 检查关键配置项
        required_keys = [
            'output_dir',
            'splatloc',
            'model',
            'dataset',
            'loss',
            'training'
        ]
        
        for key in required_keys:
            if key in config:
                print(f"  ✓ {key}")
            else:
                print(f"  ✗ 缺少配置项: {key}")
                return False
        
        # 检查SplatLoc模型路径
        print(f"\n  检查SplatLoc模型路径...")
        gaussians_path = config['splatloc']['gaussians_path']
        decoder_path = config['splatloc']['decoder_path']
        
        if os.path.exists(gaussians_path):
            size_mb = os.path.getsize(gaussians_path) / (1024 * 1024)
            print(f"    ✓ Gaussian模型: {gaussians_path} ({size_mb:.1f}MB)")
        else:
            print(f"    ✗ Gaussian模型不存在: {gaussians_path}")
            print(f"      请修改配置文件中的gaussians_path")
        
        if os.path.exists(decoder_path):
            size_mb = os.path.getsize(decoder_path) / (1024 * 1024)
            print(f"    ✓ 特征解码器: {decoder_path} ({size_mb:.1f}MB)")
        else:
            print(f"    ✗ 特征解码器不存在: {decoder_path}")
            print(f"      请修改配置文件中的decoder_path")
        
        # 检查数据集路径
        print(f"\n  检查数据集路径...")
        data_root = config['dataset']['data_root']
        train_scene = config['dataset']['train_scene']
        
        scene_path = os.path.join(data_root, train_scene)
        if os.path.exists(scene_path):
            print(f"    ✓ 训练场景: {scene_path}")
            
            # 检查RGB图像
            rgb_dir = os.path.join(scene_path, 'rgb')
            if os.path.exists(rgb_dir):
                rgb_files = list(Path(rgb_dir).glob('rgb_*.png'))
                print(f"      ✓ RGB图像: {len(rgb_files)}张")
            else:
                print(f"      ✗ RGB目录不存在: {rgb_dir}")
            
            # 检查位姿文件
            pose_file = os.path.join(scene_path, 'traj_w_c.txt')
            if os.path.exists(pose_file):
                with open(pose_file, 'r') as f:
                    num_poses = len(f.readlines())
                print(f"      ✓ 位姿文件: {num_poses}个位姿")
            else:
                print(f"      ✗ 位姿文件不存在: {pose_file}")
        else:
            print(f"    ✗ 训练场景不存在: {scene_path}")
            print(f"      请修改配置文件中的data_root和train_scene")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 配置文件解析失败: {str(e)}")
        return False

def check_gpu_memory():
    """检查GPU内存"""
    print("\n检查GPU内存...")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
            cached_memory = torch.cuda.memory_reserved(device) / (1024**3)
            free_memory = total_memory - allocated_memory
            
            print(f"  GPU: {torch.cuda.get_device_name(device)}")
            print(f"  总内存: {total_memory:.2f} GB")
            print(f"  已分配: {allocated_memory:.2f} GB")
            print(f"  已缓存: {cached_memory:.2f} GB")
            print(f"  可用: {free_memory:.2f} GB")
            
            if free_memory < 4.0:
                print(f"  ⚠ GPU内存较少，建议减小batch_size")
            else:
                print(f"  ✓ GPU内存充足")
        else:
            print(f"  ⚠ 无可用GPU")
    except Exception as e:
        print(f"  ✗ 检查失败: {str(e)}")

def main():
    """主函数"""
    print("=" * 60)
    print("隐式对应关系位姿估计 - 训练环境检查")
    print("=" * 60)
    
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"\n工作目录: {os.getcwd()}\n")
    
    checks = [
        ("Python版本", check_python_version),
        ("PyTorch", check_pytorch),
        ("依赖包", check_dependencies),
        ("项目结构", check_project_structure),
        ("配置文件", check_config_file),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n{name}检查失败: {str(e)}")
            results[name] = False
    
    # GPU内存检查（不计入结果）
    try:
        check_gpu_memory()
    except:
        pass
    
    # 总结
    print("\n" + "=" * 60)
    print("检查结果总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有检查通过！可以开始训练")
        print("\n运行以下命令开始训练：")
        print("  python train.py --config configs/train_config.yaml")
        print("或:")
        print("  ./run_train.sh")
    else:
        print("✗ 部分检查未通过，请修复上述问题后再开始训练")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
