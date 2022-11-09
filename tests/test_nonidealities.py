import pytest
import tensorflow as tf

from mnn import nonidealities


@pytest.mark.parametrize(
    "nonideality,expected_label",
    [
        (
            nonidealities.IVNonlinearityPF(
                0.5, [0.1, 0.2], [0.3, 0.4], tf.constant([[0.1, 0.5], [0.5, 0.9]])
            ),
            "IVNL_PF={m_c=0.1,c_c=0.3,m_d=0.2,c_d=0.4}",
        ),
        (
            nonidealities.StuckAt(1e-6, 0.1),
            "Stuck={value=1e-06,p=0.1}",
        ),
        (
            nonidealities.StuckAtGOff(1e-6, 0.1),
            "StuckOff={p=0.1}",
        ),
        (
            nonidealities.StuckAtGOn(5e-6, 0.1),
            "StuckOn={p=0.1}",
        ),
        (
            nonidealities.D2DLognormal(1e-6, 5e-6, 0.1, 0.05),
            "D2DLN={R_on_log_std=0.1,R_off_log_std=0.05}",
        ),
    ],
)
def test_labels(nonideality, expected_label):
    label = nonideality.label()
    assert label == expected_label
