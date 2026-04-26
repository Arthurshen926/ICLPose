#!/usr/bin/env python3
"""
构建 DINO CLS Token FAISS 检索索引
===================================
读取所有帧的 CLS token + 位姿，构建 FAISS IndexFlatIP 索引。

用法:
    PYTHONPATH=. python scripts/build_retrieval_index.py \
        --cls_dir output/features_multiscale_compressed/room_0/cls \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --save_path output/retrieval/room_0/index.faiss
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_retrieval.retrievers.cls_retrieval import PlaceRecognition


def main():
    parser = argparse.ArgumentParser(description='Build FAISS retrieval index')
    parser.add_argument('--cls_dir', type=str, required=True,
                        help='CLS token 目录')
    parser.add_argument('--traj_path', type=str, required=True,
                        help='位姿文件 (traj_w_c.txt)')
    parser.add_argument('--save_path', type=str,
                        default='output/retrieval/room_0/index.faiss',
                        help='索引保存路径')
    parser.add_argument('--dim', type=int, default=768,
                        help='CLS token 维度')
    args = parser.parse_args()

    # 构建
    retriever = PlaceRecognition.build_from_dir(
        cls_dir=args.cls_dir,
        traj_path=args.traj_path,
        dim=args.dim,
    )

    # 保存
    retriever.save(args.save_path)
    print(f"\n完成! 索引已保存至: {args.save_path}")


if __name__ == '__main__':
    main()
