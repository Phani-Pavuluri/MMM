import numpy as np

from mmm.features.design_matrix import DesignMatrixMasks, design_masks_from_cv_split


def test_design_masks_from_cv_split_shapes_and_validate():
    n = 10
    tr = np.array([True] * 6 + [False] * 4)
    va = np.array([False] * 6 + [True] * 3 + [False])
    m = design_masks_from_cv_split(n, tr, va)
    assert isinstance(m, DesignMatrixMasks)
    m.validate(n)
