"""The two-stage resampler: both stages, their intersection, and the controls."""

from __future__ import annotations

import math

import numpy as np
import pytest
from motionbloom.tremora_store.pads.p04.contract import (
    DERIVED_RATES_HZ,
    PARENT_RATE_HZ,
    RESAMPLING_RATIOS,
)
from motionbloom.tremora_store.pads.p04.controls import run_controls
from motionbloom.tremora_store.pads.p04.filters import (
    FILTER_SPECS,
    design,
    polyphase_dc_gains,
)
from motionbloom.tremora_store.pads.p04.rational_time import (
    grid_for,
    polyphase_anchor,
    supported_output_ordinals,
)
from motionbloom.tremora_store.pads.p04.resample import (
    ResampleError,
    derive_support,
    derive_window,
    window_eligibility,
    window_output_ordinals,
)
from motionbloom.tremora_store.pads.p04.stage_a import (
    PARENT_PERIOD_PS,
    StageAError,
    bracketable_parent_range,
    build_parent,
)
from motionbloom.tremora_store.pads.p04.stage_b import (
    StageBError,
    filter_to_rate,
)

WINDOW_PS = 4_000_000_000_000


def times_ps(count: int, *, start_ps: int = 0) -> np.ndarray:
    """Deterministic irregular source times, like the release's own clock."""

    deltas = np.asarray(
        [1.0e10 + 4.0e8 * math.sin(0.37 * index) for index in range(count)],
        dtype=np.float64,
    )
    return start_ps + np.rint(
        np.concatenate([[0.0], np.cumsum(deltas)[:-1]])
    ).astype(np.int64)


# --- stage A --------------------------------------------------------------


def test_the_parent_range_is_what_the_segment_can_bracket() -> None:
    source = times_ps(2048)
    parent = bracketable_parent_range(source)
    assert parent.first_ordinal * PARENT_PERIOD_PS >= source[0]
    assert parent.last_ordinal * PARENT_PERIOD_PS <= source[-1]
    # One ordinal further either way would need extrapolation.
    assert (parent.first_ordinal - 1) * PARENT_PERIOD_PS < source[0]
    assert (parent.last_ordinal + 1) * PARENT_PERIOD_PS > source[-1]


def test_an_offset_segment_does_not_start_its_parent_at_zero() -> None:
    source = times_ps(512, start_ps=7_777_777_777)
    parent = bracketable_parent_range(source)
    assert parent.first_ordinal > 0
    assert parent.first_ordinal * PARENT_PERIOD_PS >= source[0]


def test_a_segment_of_one_sample_brackets_nothing() -> None:
    assert bracketable_parent_range(np.array([0], dtype=np.int64)).empty
    assert bracketable_parent_range(np.array([], dtype=np.int64)).empty


def test_interpolation_is_exact_on_a_constant_and_on_a_ramp() -> None:
    source = times_ps(512)
    parent = bracketable_parent_range(source)
    constant = build_parent(
        source, [np.full(source.size, 2.5)],
        first_ordinal=parent.first_ordinal, last_ordinal=parent.last_ordinal,
    )
    assert np.max(np.abs(constant - 2.5)) == 0.0
    ramp = build_parent(
        source, [source.astype(np.float64)],
        first_ordinal=parent.first_ordinal, last_ordinal=parent.last_ordinal,
    )[0]
    targets = np.arange(
        parent.first_ordinal, parent.last_ordinal + 1
    ) * PARENT_PERIOD_PS
    # A linear function is reproduced by linear interpolation.
    assert np.max(np.abs(ramp - targets)) < 1e-3


def test_extrapolation_is_refused_rather_than_clamped() -> None:
    source = times_ps(256)
    parent = bracketable_parent_range(source)
    with pytest.raises(StageAError):
        build_parent(
            source, [np.ones(source.size)],
            first_ordinal=parent.first_ordinal - 1,
            last_ordinal=parent.last_ordinal,
        )
    with pytest.raises(StageAError):
        build_parent(
            source, [np.ones(source.size)],
            first_ordinal=parent.first_ordinal,
            last_ordinal=parent.last_ordinal + 1,
        )


def test_a_channel_that_does_not_match_the_times_is_refused() -> None:
    source = times_ps(64)
    parent = bracketable_parent_range(source)
    with pytest.raises(StageAError):
        build_parent(
            source, [np.ones(5)],
            first_ordinal=parent.first_ordinal,
            last_ordinal=parent.last_ordinal,
        )


# --- stage B --------------------------------------------------------------


@pytest.mark.parametrize("rate", (50, 25))
def test_integer_decimation_matches_a_direct_convolution(rate: int) -> None:
    taps = design(rate)
    delay = (taps.size - 1) // 2
    parent = np.sin(np.arange(2000) * 0.05)[None, :]
    ordinals = supported_output_ordinals(
        rate, taps=taps.size, parent_first=0, parent_last=1999
    )
    produced = filter_to_rate(
        parent, rate_hz=rate, parent_first_ordinal=0,
        output_ordinals=ordinals,
    )[0]
    decimate = RESAMPLING_RATIOS[rate][1]
    expected = np.array([
        float(np.dot(
            taps,
            parent[0][decimate * k + delay - np.arange(taps.size)],
        ))
        for k in ordinals
    ])
    assert np.allclose(produced, expected, rtol=0, atol=1e-12)


def test_the_polyphase_path_matches_upsample_filter_decimate() -> None:
    taps = design(30)
    upsample, decimate = RESAMPLING_RATIOS[30]
    delay = (taps.size - 1) // 2
    parent = np.cos(np.arange(1200) * 0.03)[None, :]
    ordinals = supported_output_ordinals(
        30, taps=taps.size, parent_first=0, parent_last=1199
    )
    produced = filter_to_rate(
        parent, rate_hz=30, parent_first_ordinal=0,
        output_ordinals=ordinals,
    )[0]
    # The literal definition: zero-stuff, convolve, take every tenth sample.
    stuffed = np.zeros(parent.shape[1] * upsample)
    stuffed[::upsample] = parent[0]
    expected = np.array([
        float(np.dot(
            taps,
            stuffed[decimate * k + delay - np.arange(taps.size)],
        ))
        for k in ordinals
    ])
    assert np.allclose(produced, expected, rtol=0, atol=1e-12)


def test_filtering_beyond_the_parent_is_refused_not_padded() -> None:
    parent = np.ones((1, 100))
    with pytest.raises(StageBError):
        filter_to_rate(
            parent, rate_hz=25, parent_first_ordinal=0,
            output_ordinals=range(3),
        )


# --- the support intersection ---------------------------------------------


@pytest.mark.parametrize("rate", (50, 30, 25))
def test_the_exact_support_bound_matches_a_brute_force_scan(
    rate: int,
) -> None:
    taps = design(rate).size
    for first, last in ((0, 1999), (37, 1490), (500, 900)):
        computed = supported_output_ordinals(
            rate, taps=taps, parent_first=first, parent_last=last
        )
        brute = []
        for ordinal in range((last * rate) // PARENT_RATE_HZ + 4):
            _, anchor, branch = polyphase_anchor(rate, ordinal, taps=taps)
            if anchor - branch + 1 >= first and anchor <= last:
                brute.append(ordinal)
        assert list(computed) == brute


def test_an_unbracketable_parent_end_propagates_into_every_rate() -> None:
    # A segment starting late in task-local time cannot bracket the early
    # parent ordinals, and that absence must reach the derived mask before
    # the filter guard is applied.
    source = times_ps(2048, start_ps=5_000_000_000_000)
    parent = bracketable_parent_range(source)
    assert parent.first_ordinal >= 500
    for rate in DERIVED_RATES_HZ:
        support = derive_support(source, rate)
        assert support.parent.first_ordinal == parent.first_ordinal
        assert support.first_time_ps is not None
        assert support.first_time_ps >= source[0]
        if rate in FILTER_SPECS:
            # The filter guard sits strictly inside the bracketable region.
            assert support.first_time_ps > parent.first_ordinal * (
                PARENT_PERIOD_PS
            )


@pytest.mark.parametrize("rate", DERIVED_RATES_HZ)
def test_lower_rates_lose_more_of_the_segment_to_their_kernels(
    rate: int,
) -> None:
    source = times_ps(2048)
    support = derive_support(source, rate)
    assert support.supported
    span = (support.last_time_ps or 0) - (support.first_time_ps or 0)
    assert span < source[-1] - source[0] or rate == PARENT_RATE_HZ


def test_the_parent_rate_carries_no_filter_guard() -> None:
    source = times_ps(1024)
    support = derive_support(source, PARENT_RATE_HZ)
    assert list(support.supported) == list(support.parent.as_range())


def test_a_segment_shorter_than_the_kernel_supports_nothing() -> None:
    source = times_ps(40)
    for rate in (30, 25):
        assert not derive_support(source, rate).supported


# --- window eligibility ---------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "expected"), ((100, 400), (50, 200), (30, 120), (25, 100))
)
def test_a_four_second_window_holds_exactly_the_grid_count(
    rate: int, expected: int
) -> None:
    source = times_ps(2048)
    support = derive_support(source, rate)
    start = (support.first_time_ps or 0) + 1_000_000_000_000
    status, ordinals = window_eligibility(support, start, start + WINDOW_PS)
    assert status == "WINDOW_ELIGIBLE"
    assert len(ordinals) == expected
    values = derive_window(
        source, [np.ones(source.size)], rate_hz=rate,
        support=support, ordinals=ordinals,
    )
    assert values.shape == (1, expected)


def test_a_window_reaching_outside_supported_output_is_refused() -> None:
    source = times_ps(2048)
    for rate in (50, 30, 25):
        support = derive_support(source, rate)
        status, ordinals = window_eligibility(support, 0, WINDOW_PS)
        assert status == "WINDOW_OUTSIDE_SUPPORT"
        assert not ordinals
        with pytest.raises(ResampleError):
            derive_window(
                source, [np.ones(source.size)], rate_hz=rate,
                support=support, ordinals=range(4),
            )


def test_window_ordinals_use_a_half_open_interval() -> None:
    grid = grid_for(50)
    ordinals = window_output_ordinals(50, 0, WINDOW_PS)
    assert len(ordinals) == 200
    assert grid.sample_picoseconds(ordinals.stop - 1) < WINDOW_PS
    assert grid.sample_picoseconds(ordinals.stop) == WINDOW_PS


# --- the in-process controls ---------------------------------------------


@pytest.fixture(scope="module")
def controls() -> dict:
    return run_controls()


def test_every_resampling_control_passes(controls: dict) -> None:
    assert controls["status"] == "RESAMPLING_CONTROLS_PASS"
    assert controls["controls_passed"] == controls["controls_total"] == 11


def test_the_constant_input_control_pins_the_branch_gains(
    controls: dict,
) -> None:
    measured = controls["measured"]["constant_input_30"]
    published = polyphase_dc_gains(30)
    assert measured["mean"] == pytest.approx(1.0, abs=1e-6)
    # The realized per-output ripple *is* the published branch spread.
    assert measured["observed_ripple"] == pytest.approx(
        max(published) - min(published), abs=1e-12
    )
    # And within a phase the gain is bit-identical, so nothing is being
    # renormalized per output.
    assert measured["within_phase_spread"] == 0.0
    for rate in (100, 50, 25):
        assert controls["measured"][f"constant_input_{rate}"][
            "observed_ripple"
        ] == 0.0


def test_the_controls_show_no_out_of_band_tone_folding_in(
    controls: dict,
) -> None:
    assert controls["controls"][
        "out_of_band_tone_does_not_fold_into_the_band"
    ]
    # 18 Hz would fold to 12 Hz at 30 Hz; it does not survive.
    folded = controls["measured"]["alias_30"]
    assert folded["folds_to_hz"] == 12.0
    assert folded["db"] < -60.0


def test_the_controls_show_the_preservation_band_surviving(
    controls: dict,
) -> None:
    for rate in DERIVED_RATES_HZ:
        entry = controls["measured"][f"preserved_{rate}"]
        assert entry["amplitude"] > 0.9
    # The dominant in-band cost is stage A's uniformization, not the derived
    # filters: 50/30/25 track the 100 Hz parent closely.
    parent = controls["measured"]["preserved_100"]["db"]
    for rate in (50, 30):
        assert abs(
            controls["measured"][f"preserved_{rate}"]["db"] - parent
        ) < 0.02
