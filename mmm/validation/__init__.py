"""Validation utilities.

Package initializer intentionally avoids eager imports of synthetic validators,
certification runners, or model-dependent modules.

Import concrete validators directly from their modules, e.g.
``mmm.validation.synthetic.hierarchy_evidence_validator``.
"""

from mmm.validation.continuous_validation import build_continuous_validation_report
from mmm.validation.decision_validation import build_decision_validation_report

__all__ = [
    "build_continuous_validation_report",
    "build_decision_validation_report",
]
