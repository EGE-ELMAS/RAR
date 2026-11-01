"""Supervised relation-extraction baselines."""

from .features import RELATION_LABELS, PureFeatureBuilder, label_to_id

__all__ = ["RELATION_LABELS", "PureFeatureBuilder", "label_to_id"]
