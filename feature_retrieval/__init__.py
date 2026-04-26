"""FeatureRetrieval system facade."""

from feature_retrieval.retrievers.cls_retrieval import PlaceRecognition
from feature_retrieval.retrievers.vlad_retrieval import VLADEncoder, VLADPlaceRecognition


__all__ = [
    "PlaceRecognition",
    "VLADEncoder",
    "VLADPlaceRecognition",
]
