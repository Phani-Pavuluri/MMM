from mmm.config.extensions import OptimizationGateConfig
from mmm.optimization.safety_gate import OptimizationSafetyGate


def test_gate_disabled_allows():
    g = OptimizationSafetyGate(OptimizationGateConfig(enabled=False))
    assert g.check(governance={}, response_diag=None, identifiability_score=1.0).allowed


def test_gate_blocks_bad_identifiability():
    cfg = OptimizationGateConfig(enabled=True, allow_missing_extension_report=False)
    g = OptimizationSafetyGate(cfg)
    r = g.check(
        governance={"approved_for_optimization": True},
        response_diag={"safe_for_optimization": True},
        identifiability_score=0.99,
    )
    assert not r.allowed
