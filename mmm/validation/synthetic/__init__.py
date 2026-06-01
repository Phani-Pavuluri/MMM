"""Synthetic validation utilities.

This package initializer intentionally avoids eager imports to prevent circular
imports during core validation/model initialization.

Import concrete runners directly from their modules, e.g.
``mmm.validation.synthetic.hierarchy_evidence_validator``.
"""

__all__: list[str] = []
