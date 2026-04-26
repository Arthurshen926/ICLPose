"""Legacy compatibility facade for FeatureRetrieval.

Active code should import from feature_retrieval.*.
"""

from feature_retrieval.retrievers.cls_retrieval import PlaceRecognition
from feature_retrieval.retrievers.vlad_retrieval import VLADEncoder, VLADPlaceRecognition


__all__ = ["PlaceRecognition", "VLADEncoder", "VLADPlaceRecognition"]
