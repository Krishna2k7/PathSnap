"""
HMM package for PathSnap sequential route reconstruction.
Exposes NetworkTransitionModel and ViterbiDecoder.
"""

from src.hmm.transition import NetworkTransitionModel
from src.hmm.viterbi import ViterbiDecoder, compute_path_continuity

__all__ = [
    "NetworkTransitionModel",
    "ViterbiDecoder",
    "compute_path_continuity",
]
