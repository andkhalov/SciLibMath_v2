"""Fuzzy membership functions for linguistic variables.
Ref: MATH.md M.6.1, M.6.2

Linguistic terms: LO (Low), MD (Medium), HI (High)
                  DEC (Decreasing), STB (Stable), INC (Increasing)

Gaussian membership: μ(x; c, σ) = exp(-(x-c)²/(2σ²))
"""

import torch
import math


def gaussian_mf(x: torch.Tensor, center: float, sigma: float) -> torch.Tensor:
    """Gaussian membership function μ(x; c, σ) = exp(-(x-c)²/(2σ²))."""
    return torch.exp(-0.5 * ((x - center) / sigma) ** 2)


class FuzzyVariable:
    """A linguistic variable with named fuzzy terms.

    Each term is a Gaussian MF with (center, sigma).
    """

    def __init__(self, name: str, terms: dict[str, tuple[float, float]]):
        """
        Args:
            name: variable name (e.g., "L_t", "EMA_m")
            terms: {term_name: (center, sigma)} for each linguistic term
        """
        self.name = name
        self.terms = terms

    def fuzzify(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute membership degree for each term."""
        return {name: gaussian_mf(x, c, s) for name, (c, s) in self.terms.items()}


# Standard linguistic variables (MATH.md M.6.2)
# Calibrated for z-score normalized inputs: values ~[-3, 3], mean≈0, std≈1.
# Raw s_t is normalized in TSFuzzyController._normalize_state() before fuzzification.

# Loss magnitude: LO/MD/HI
def loss_variable(name: str = "L") -> FuzzyVariable:
    return FuzzyVariable(name, {
        "LO": (-1.0, 0.5),   # below average
        "MD": (0.0, 0.4),    # near average
        "HI": (1.0, 0.5),    # above average
    })


# EMA trend: DEC/STB/INC
def trend_variable(name: str = "EMA") -> FuzzyVariable:
    return FuzzyVariable(name, {
        "DEC": (-1.0, 0.5),  # decreasing
        "STB": (0.0, 0.3),   # stable
        "INC": (1.0, 0.5),   # increasing
    })


# Variance: LO/HI
def variance_variable(name: str = "Var") -> FuzzyVariable:
    return FuzzyVariable(name, {
        "LO": (-0.5, 0.5),   # low variance
        "HI": (1.0, 0.5),    # high variance
    })


# Collapse indicator: LO/HI
def collapse_variable(name: str = "collapse") -> FuzzyVariable:
    return FuzzyVariable(name, {
        "LO": (-0.5, 0.5),
        "HI": (1.0, 0.5),
    })
