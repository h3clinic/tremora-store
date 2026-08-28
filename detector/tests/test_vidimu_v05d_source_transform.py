from __future__ import annotations

import hashlib
import os
import tempfile
import zipfile
from pathlib import Path

import pytest
from motionbloom.tremora_store.v05d import authority
from motionbloom.tremora_store.v05d import source_transform as transform

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = (
    REPO_ROOT
    / "data/snapshots/vidimu"
    / "a6e2194aee5478718e6f92cf9306214e361b08bb61363998f1e6e59e7378f1eb"
)
V05_SCRIPT = REPO_ROOT / "detector/benchmarks/audit_vidimu_v05_sync_authority.py"
V05_REPORT = (
    REPO_ROOT / "detector/benchmarks/vidimu_v05_sync_authority_audit.json"
)


def _external(name: str) -> Path:
    value = os.environ.get(name)
    if value is None:
        pytest.skip(f"release mode requires {name}")
    path = Path(value)
    if not path.is_file():
        pytest.fail(f"release input is missing: {name}")
    return path


@pytest.fixture(scope="module")
def real_evidence() -> dict[str, object]:
    if os.environ.get("VIDIMU_V05D_REAL_INPUTS") != "1":
        pytest.skip("set VIDIMU_V05D_REAL_INPUTS=1 for the pinned release proof")
    return transform.audit_source_transform_evidence(
        SNAPSHOT,
        _external("VIDIMU_V05D_ANALYSIS_ARCHIVE"),
        _external("VIDIMU_V05D_TOOLS_ARCHIVE"),
        V05_SCRIPT,
        V05_REPORT,
    )


def test_integer_truncation_matches_upstream() -> None:
    assert [transform.source_removed_rows("csv", cut) for cut in range(1, 15)] == (
        list(range(1, 15))
    )
    assert [transform.source_removed_rows("mot", cut) for cut in range(1, 15)] == [
        1, 3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 23
    ]
    assert [transform.source_removed_rows("raw", cut) for cut in range(1, 15)] == [
        8, 16, 25, 33, 41, 50, 58, 66, 75, 83, 91, 100, 108, 116
    ]
    with pytest.raises(transform.SourceTransformError):
        transform.source_removed_rows("mp4", 1)


def test_source_trim_operation_reproduced() -> None:
    source = b"header\r\nrow-0\r\nrow-1\r\nrow-2\r\nrow-3\r\n"
    assert transform.apply_source_trim(
        source, retained_prefix_lines=1, removed_rows=2
    ) == b"header\r\nrow-2\r\nrow-3\r\n"

    mot = b"".join(
        [*(f"header-{index}\r\n".encode() for index in range(8))]
        + [*(f"dynamic-{index}\r\n".encode() for index in range(5))]
    )
    assert transform.apply_source_trim(
        mot, retained_prefix_lines=8, removed_rows=3
    ).endswith(b"dynamic-3\r\ndynamic-4\r\n")


def test_incomplete_sensor_tick_rejected() -> None:
    expected = ("a", "b", "c", "d", "e")
    incomplete = transform.classify_raw_sensor_group(("a", "b"), expected)
    duplicate = transform.classify_raw_sensor_group(
        ("a", "b", "c", "d", "d"), expected
    )
    unknown = transform.classify_raw_sensor_group(
        ("a", "b", "c", "d", "unknown"), expected
    )
    malformed = transform.classify_raw_sensor_group(
        ("b", "a", "c", "d", "e"), expected
    )
    assert incomplete.group_status == "INCOMPLETE"
    assert duplicate.group_status == "DUPLICATE_SENSOR"
    assert unknown.group_status == "UNKNOWN_SENSOR"
    assert malformed.group_status == "MALFORMED"
    assert all(
        value.timing_authority == "NONE_RAW_POLL_GROUP_ONLY"
        for value in (incomplete, duplicate, unknown, malformed)
    )


def test_ordinal_identity_survives_trimming() -> None:
    # MOT physical lines 1--8 include metadata, columns, and the N-pose row.
    source = b"".join(
        [*(f"retained-{index}\r\n".encode() for index in range(8))]
        + [*(f"dynamic-{index}\r\n".encode() for index in range(8))]
    )
    removed = transform.source_removed_rows("mot", 2)
    result = transform.apply_source_trim(
        source, retained_prefix_lines=8, removed_rows=removed
    )
    assert result.splitlines()[8] == b"dynamic-3"


def test_source_notebook_hashes_are_frozen() -> None:
    assert authority.ESTIMATE_NOTEBOOK_SHA256 == (
        "a88c3bb86a27587ca30d99f820643bb48e27500e58f62b035174143e9c1e4865"
    )
    assert authority.MODIFY_NOTEBOOK_SHA256 == (
        "5b719a4e6b80419df18f0711dc62f44e0d7cbdb6bb4337847f3281be097d5fbf"
    )
    assert authority.SYNC_UTILITY_SHA256 == (
        "f2674ded71f19a837c9e7cb5f6678ae7944ad8f09932e029845d0766b55f139d"
    )


def test_pinned_descriptor_survives_path_replacement() -> None:
    original = b"original-pinned-bytes"
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.zip"
        replacement = root / "replacement.zip"
        source.write_bytes(original)
        replacement.write_bytes(b"untrusted-replacement")
        with transform._verified_regular_snapshot(
            source,
            maximum_bytes=1024,
            expected_sha256=hashlib.sha256(original).hexdigest(),
        ) as handle:
            os.replace(replacement, source)
            assert handle.read() == original
        assert source.read_bytes() == b"untrusted-replacement"


def test_verified_snapshot_survives_later_in_place_mutation() -> None:
    original = b"original-pinned-bytes"
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "source.zip"
        source.write_bytes(original)
        with (
            transform._verified_regular_snapshot(
                source,
                maximum_bytes=1024,
                expected_sha256=hashlib.sha256(original).hexdigest(),
            ) as handle,
            source.open("r+b") as mutator,
        ):
            mutator.seek(0)
            mutator.write(b"mutated!")
            mutator.flush()
            os.fsync(mutator.fileno())
            assert handle.read() == original


def test_all_217_overrides_bound_to_source_rows(
    real_evidence: dict[str, object],
) -> None:
    instructions = real_evidence["source_instructions"]
    trim = real_evidence["source_trim_reproduction"]
    assert instructions["info_row_count"] == 366
    assert instructions["non_mp4_instruction_count"] == 217
    assert instructions["non_mp4_instruction_recording_count"] == 181
    assert trim["overrides_expected"] == 217
    assert trim["overrides_bound"] == 217


def test_override_row_hashes_are_stable(
    real_evidence: dict[str, object],
) -> None:
    instructions = real_evidence["source_instructions"]
    assert instructions["info_to_sync_sha256"] == (
        "4596803e90e2908717ab227846f7a1c10a5c6b29c9bcbddb90e940a7259c519f"
    )
    assert instructions["non_mp4_instruction_manifest_sha256"] == (
        "56f82dbf359ad4bbed3a5df70a0f81c28c58ced16df1248cdd0e3bc6b047fb2c"
    )
    assert instructions["override_row_ordinal_basis"] == (
        "ZERO_BASED_DATA_ROW_SOURCE_ORDER"
    )
    assert instructions["override_row_sha256_basis"] == (
        "EXACT_SOURCE_ROW_BYTES_INCLUDING_CRLF"
    )


def test_source_trim_operation_reproduced_for_all_published_derivatives(
    real_evidence: dict[str, object],
) -> None:
    trim = real_evidence["source_trim_reproduction"]
    assert trim["overrides_reproduced"] == 217
    assert trim["overrides_unreproduced"] == 0
    assert trim["published_derivative_directory_count"] == 17
    assert trim["all_generated_derivatives_byte_identical"] is True
    assert trim["discrepancies"] == []
    assert trim["source_trim_overlay_manifest_sha256"] == (
        "e35562327b33cff2a74251df7a5ba7e08fd0cc345550cc9d67d5bdd8eedc1600"
    )


def test_raw_rows_grouped_into_five_sensor_ticks(
    real_evidence: dict[str, object],
) -> None:
    # This required test intentionally disproves the proposed tick semantics.
    raw = real_evidence["raw_poll_to_sto_mot_reconciliation"]
    assert raw["all_original_raw_groups_structurally_complete"] is True
    assert raw["raw_dynamic_poll_groups"] == 2_036_601
    assert raw["sto_dynamic_ordinal_rows"] == 299_711
    assert raw["mot_dynamic_ordinal_rows"] == 299_711
    assert raw["raw_groups_are_nominal_50hz_ticks"] is False
    assert raw["raw_group_timing_authority"] == "NONE_RAW_POLL_GROUP_ONLY"
    assert raw["aggregate_raw_group_to_sto_dynamic_ratio"] == "6.795216"
    assert raw["per_record_ratio_minimum"] == "5.261773"
    assert raw["per_record_ratio_median"] == "6.562073"
    assert raw["per_record_ratio_maximum"] == "7.865480"


def test_source_tool_members_bound_in_release_mode() -> None:
    if os.environ.get("VIDIMU_V05D_REAL_INPUTS") != "1":
        pytest.skip("set VIDIMU_V05D_REAL_INPUTS=1 for the pinned release proof")
    tools = _external("VIDIMU_V05D_TOOLS_ARCHIVE")
    with zipfile.ZipFile(tools) as archive:
        assert archive.comment == authority.SOURCE_TOOLS_COMMIT.encode("ascii")
    pins = transform.verify_frozen_inputs(
        SNAPSHOT,
        _external("VIDIMU_V05D_ANALYSIS_ARCHIVE"),
        tools,
        V05_SCRIPT,
        V05_REPORT,
    )
    assert pins["critical_source_tool_member_hashes"] == {
        "synchronize/EstimateFileSynchronization.ipynb": (
            "a88c3bb86a27587ca30d99f820643bb48e27500e58f62b035174143e9c1e4865"
        ),
        "synchronize/ModifyFilesToSync.ipynb": (
            "5b719a4e6b80419df18f0711dc62f44e0d7cbdb6bb4337847f3281be097d5fbf"
        ),
        "utils/fileProcessing.py": (
            "2b534daa9887934824e0034ce7af42414ab1884743a581a598300949693f4331"
        ),
        "utils/signalProcessing.py": (
            "88e4701b71e7b7c048f64f1698220d22ee0887b97624aa2c38fae8d37861c591"
        ),
        "utils/syncUtilities.py": (
            "f2674ded71f19a837c9e7cb5f6678ae7944ad8f09932e029845d0766b55f139d"
        ),
    }
