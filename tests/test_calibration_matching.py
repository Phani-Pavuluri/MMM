from mmm.calibration.matching import match_experiments
from mmm.calibration.schema import ExperimentObservation


def test_match_filters_channel():
    exps = [
        ExperimentObservation(
            experiment_id="e1", geo_id="G0", channel="search", lift=0.1, lift_se=0.02
        ),
        ExperimentObservation(
            experiment_id="e2", geo_id="G0", channel="unknown", lift=0.1, lift_se=None
        ),
    ]
    m = match_experiments(exps, available_geos={"G0"}, available_channels={"search"}, match_levels=["geo"])
    assert len(m) == 1
