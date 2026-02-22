"""
Place Recognition Module
=========================
场景检索/地点重识别模块。

包含两种检索方案:
  1. CLS Token (768d): 基于 DINO CLS token 的快速全局检索
  2. VLAD (K×768d): 基于 DINOv2 patch tokens + VLAD 聚合的高质量检索

核心组件:
  - cls_retrieval.py:         PlaceRecognition (CLS Token + FAISS)
  - vlad_retrieval.py:        VLADPlaceRecognition (VLAD + FAISS)
  - extract_cls.py:           CLS token 提取脚本
  - extract_dino_patches.py:  DINO patch token 提取脚本
  - build_index.py:           构建 FAISS 检索索引
  - eval_retrieval.py:        CLS vs VLAD 对比评估
  - eval_cross_sequence.py:   跨序列检索评估
  - eval_self_retrieval.py:   自检索验证
"""

from place_recognition.cls_retrieval import PlaceRecognition
from place_recognition.vlad_retrieval import VLADPlaceRecognition, VLADEncoder
