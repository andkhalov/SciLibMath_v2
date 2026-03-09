"""Visual alignment loss wrapper.
Ref: MATH.md M.1.2*, M.3.0

L_visual_align is computed by AlignNet (models/align_net.py).
This module just provides a consistent interface.
In E1-E5: lambda_va = const (static).
In E6-E7: lambda_va controlled by fuzzy controller.
"""
# Visual align loss is computed directly by FamilyA.forward() → visual_align_loss.
# No separate module needed here — the AlignNet in models/ handles it.
# This file exists for completeness and future extension.
