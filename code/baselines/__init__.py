"""Multi-task balancing baselines for comparison with T-S fuzzy controller (E8c).

BL1: GradNorm (Chen et al. ICML 2018)
BL2: PCGrad (Yu et al. NeurIPS 2020)
BL3: Uncertainty Weighting (Kendall et al. CVPR 2018)
"""

from .gradnorm import GradNormBalancer
from .pcgrad import PCGradOptimizer
from .uncertainty_weighting import UncertaintyWeighting
