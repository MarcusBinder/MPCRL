import numpy as np
import pytest

pytest.importorskip("py_wake")

# PyWake uses meteorological convention (degrees from which wind originates).
# For our layout, wind coming from the west (left) corresponds to wd=270°.

from nmpc_windfarm_acados_fixed import build_pywake_model


DEFAULT_WIND_DIR = 270.0  # wind coming from positive y-axis (from the left)


def _simulate_powers(wf_model, layout, yaw_deg, wd_deg=DEFAULT_WIND_DIR):
    """Run PyWake for a single wind direction and return per-turbine power in MW."""
    x = layout["x"]
    y = layout["y"]
    yaw_vector = np.asarray(yaw_deg, dtype=float)
    if yaw_vector.ndim == 0:
        yaw_vector = np.full_like(x, yaw_vector, dtype=float)
    yaw_ilk = yaw_vector.reshape(len(x), 1, 1)
    sim = wf_model(x=x, y=y, wd=[wd_deg], ws=[8.0], yaw=yaw_ilk, tilt=0)
    return sim.Power.values[:, 0, 0] / 1e6


def test_downstream_turbine_experiences_wake_loss():
    """Aligned turbines should show a meaningful downstream power deficit."""
    D = 178.0
    spacing = 5 * D
    x = np.array([0.0, spacing])
    y = np.array([0.0, 0.0])
    wf_model, layout = build_pywake_model(x, y, D)

    power_aligned = _simulate_powers(wf_model, layout, np.zeros(2))

    downstream_loss_ratio = power_aligned[1] / power_aligned[0]
    assert downstream_loss_ratio < 0.9, (
        f"Expected at least 10% downstream wake loss, "
        f"got ratio={downstream_loss_ratio:.3f}"
    )


def test_upstream_yaw_recovers_downstream_power():
    """Yawing the upstream turbine should increase downstream power while reducing upstream power."""
    D = 178.0
    spacing = 5 * D
    x = np.array([0.0, spacing])
    y = np.array([0.0, 0.0])
    wf_model, layout = build_pywake_model(x, y, D)

    power_aligned = _simulate_powers(wf_model, layout, np.zeros(2))
    power_yawed = _simulate_powers(wf_model, layout, np.array([25.0, 0.0]))

    upstream_change = power_yawed[0] - power_aligned[0]
    downstream_change = power_yawed[1] - power_aligned[1]

    assert upstream_change < -0.05, (
        f"Expected upstream cosine loss of at least 0.05 MW, "
        f"got change={upstream_change:.3f} MW"
    )
    assert downstream_change > 0.05, (
        f"Expected downstream gain of at least 0.05 MW, "
        f"got change={downstream_change:.3f} MW"
    )


def test_optimal_pattern_for_6d_spacing_matches_expected_yaws():
    """
    For 6D spacing at 10 m/s and TI≈0.06 the classic wake-steering optimum is
    ~20° yaw on upstream turbines and ~0° on the last turbine.
    """
    D = 178.0
    spacing = 6 * D
    x = np.array([0.0, spacing, 2 * spacing, 3 * spacing])
    y = np.zeros_like(x)
    wf_model, layout = build_pywake_model(x, y, D, ti=0.06)

    cases = {
        "aligned": np.array([0.0, 0.0, 0.0, 0.0]),
        "yaw10": np.array([10.0, 10.0, 10.0, 0.0]),
        "yaw15": np.array([15.0, 15.0, 15.0, 0.0]),
        "yaw20": np.array([20.0, 20.0, 20.0, 0.0]),
        "yaw25": np.array([25.0, 25.0, 25.0, 0.0]),
        "yaw20_last5": np.array([20.0, 20.0, 20.0, 5.0]),
    }

    powers = {
        name: _simulate_powers(wf_model, layout, yaw, wd_deg=270.0).sum()
        for name, yaw in cases.items()
    }

    assert powers["yaw20"] > powers["aligned"] + 1.0, "Yawing should deliver clear net gain"

    best_name = max(powers, key=powers.get)
    assert best_name in {"yaw15", "yaw20", "yaw25"}, (
        f"Unexpected optimum among discrete candidates: best={best_name}, powers={powers}"
    )

    # Ensure the 20° case is within a narrow margin of the best and better than
    # adding yaw to the downstream machine.
    assert powers["yaw20"] >= powers["yaw20_last5"] + 0.01, (
        "Downstream turbine should prefer near-zero yaw when upstream are yawed"
    )
    assert abs(powers["yaw20"] - powers[best_name]) <= 0.05, (
        f"Best candidate deviates >0.05 MW from 20° case: {powers}"
    )
