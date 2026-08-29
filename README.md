# TremoraStore

TremoraStore is the public-data systems layer for provenance-bound video–IMU
storage, deterministic replay, and synchronization-authority auditing. This is
a code-and-evidence release; it contains no participant recordings, clinical
validation data, or redistributed VIDIMU source archives.

## Current evidence boundary

- VIDIMU v0.4 Gate B: deterministic source-to-CV finalization `PASS` in the
  frozen observed environment.
- VIDIMU v0.5: audit `PASS`; raw native-clock authority
  `NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`.
- VIDIMU v0.5D: audit `PASS`; source-derived alignment materialization
  `NO_GO`.
- No canonical clocks, frame-to-IMU indexes, multimodal windows, alignment
  Parquet tables, or synchronization success markers are claimed by v0.5 or
  v0.5D.
- PADS-P0.1: ingest audit `PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK`, 14/14
  conditions against the published PhysioNet release, reproduced across two
  processes.
- PADS-P0.2: index and window audit `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY`,
  16/16 conditions; 13,447,168 samples stored exactly once, 50,676 gap-aware
  four-second windows, and byte-exact replay of all 10,318 streams read back
  from the store. No spectrum, resampled signal or benchmark result is
  claimed.
- E4D-P0.1: machinery only. Ego4D needs a signed licence and pinned assets, so
  the audit has never run against Ego4D data -- the CLI returns
  `BLOCKED_INPUT_DATA_UNAVAILABLE` and exits 4.

The v0.5D audit reproduced all 217 published non-MP4 source transformations
byte-for-byte. It withheld materialization because the released evidence does
not authorize a RAW-poll-group-to-STO/MOT ordinal map or a CSV-to-decoded-frame
mapping. See the [checked audit](detector/benchmarks/vidimu_v05d_derived_alignment_release_audit.json)
and [benchmark specification](detector/docs/tremora-store/TremoraStore_Public_Data_Benchmark_Spec_v0_1.md).

## Repository layout

```text
detector/
├── motionbloom/tremora_store/   # storage, replay, source, CV, and authority code
├── benchmarks/                  # checked release audits and audit commands
├── tests/                       # VIDIMU and fail-closed regression tests
└── docs/tremora-store/          # authoritative public scope and artifact status
```

## Run the code-only tests

Python 3.12 is the reproducible public CI target.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=detector python -m pytest -q detector/tests/test_vidimu*.py
```

Tests requiring the pinned VIDIMU archives or frozen model/runtime inputs are
release-gated and skip when those external inputs are absent. The source data
and model binary are deliberately not stored in this repository.

## Provenance

This public snapshot corresponds to the following bounded commits in the
private development monorepo:

- `06409da73f1c80200bee9bce64fbf2823f61051c`
- `5760b45db0f7502b7a55ac808cce35c53e35aeea`
- `658be1c8183254225124a6ab2803cdf63142fe69`

The checked v0.5D audit SHA-256 is
`131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0`.

## License and attribution

TremoraStore code is released under the [MIT License](LICENSE). VIDIMU dataset
and source-tool attribution is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
