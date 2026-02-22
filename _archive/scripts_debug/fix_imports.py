#!/usr/bin/env python3
"""
修复splatloc_modules中的所有内部导入路径
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """修复单个文件中的导入"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # 修复模式
    patterns = [
        (r'from gaussian_splatting\.', 'from splatloc_modules.gaussian_splatting.'),
        (r'from scene\.', 'from splatloc_modules.gaussian_splatting.scene.'),
        (r'from utils\.', 'from splatloc_modules.gaussian_splatting.utils.'),
        (r'import gaussian_splatting\.', 'import splatloc_modules.gaussian_splatting.'),
        (r'from models\.', 'from splatloc_modules.models.'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    root = Path('/home/yons/Projects/implicit_correspondence/splatloc_modules')
    
    fixed_files = []
    for pyfile in root.rglob('*.py'):
        if fix_imports_in_file(pyfile):
            fixed_files.append(pyfile)
            print(f'✓ Fixed: {pyfile.relative_to(root)}')
    
    print(f'\n总共修复了 {len(fixed_files)} 个文件')

if __name__ == '__main__':
    main()
