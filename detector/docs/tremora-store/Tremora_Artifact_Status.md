# Tremora artifact status

**Status date:** 2026-08-29  
**Authority:** This registry records scope status without modifying historical
artifacts.

| Artifact | Status | SHA-256 |
|---|---|---|
| `TremoraStore_Public_Data_Benchmark_Spec_v0_1.md` | **Current authoritative scope**: public-data-only storage, indexing, synchronization, replay, and regeneration-verified completeness benchmark; VIDIMU v0.4 Gate B is `PASS`, while both the v0.5 RAW-native authority gate and v0.5D source-derived materialization gate are `NO-GO`; v0.6/v0.6D remain closed; PADS-P0.1 ingest and PADS-P0.2 indexing are both `PASS`, and E4D-P0.1 is implemented but unevaluated | `a239e8cf7b99f2b062e5de0683c0742b5e7c5b2259f040363125ec30ca0e1640` |
| `TremoraStore_Dataset_Architecture_v0_2.md` | **Current dataset-role and timing-authority scope**: supplements the v0.1 specification; freezes the eight timing-authority tiers and the three-dataset architecture, and records the E4D-P0.1, PADS-P0.1 and PADS-P0.2 contracts | `1d0c4388f76dace211b1b3c6dcd49153d68415af56dcca3a0624c65175dadf71` |
| `fullmotion/detector/benchmarks/vidimu_v2_release_audit.json` | **Current implementation evidence**: deterministic `PASS` audit of all 208 pinned VIDIMU CSV/RAW pairs plus central-directory-only original-video candidate inventory | `41277661f9e248da2f42c0703b69beec92bcaf0037b5d46264f64852ab22ecf1` |
| `fullmotion/detector/benchmarks/vidimu_v04_gate_b_release_audit.json` | **Current Gate-B empirical evidence**: `BYTE_IDENTICAL_SOURCE_TO_CV_PASS` / `PASS` for two clean, process/root/inode-disjoint all-208 executions in the frozen observed environment | `d24863d20347cf2c9ab092a9f7771ada3a88ec8fbc77a7b33788df5c0637a10e` |
| `fullmotion/detector/benchmarks/vidimu_v05_sync_authority_audit.json` | **Current v0.5 authority evidence**: audit execution `PASS`, gate `NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`; all 208 original RAW assets and all 217 released synchronization overrides reconcile, two records are `AMBIGUOUS_SOURCE_MAPPING`, and no canonical clocks or v0.6 artifacts were emitted | `3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec` |
| `fullmotion/detector/benchmarks/vidimu_v05d_derived_alignment_release_audit.json` | **Current v0.5D evidence**: audit execution `PASS`, source-derived materialization gate `NO_GO`; all 217 source transformations reproduce byte-for-byte, but 2,036,601 RAW polling groups do not map one-to-one to 299,711 50 Hz STO/MOT ordinals, 30/34 RAW trims end mid group, and no alignment Parquet or success marker was emitted | `131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0` |
| `fullmotion/detector/benchmarks/pads_p01_release_audit.json` | **Current PADS-P0.1 empirical evidence**: audit execution `PASS`, gate `PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK`, 14/14 conditions; 469 participants, 5,159 assessment steps and 10,318 device files reconciled, 11,256 files hash-verified against the release's own `SHA256SUMS.txt`, 13,447,168 samples parsed, evidence hash `e25ce02f…` reproduced by two inode-disjoint processes | `6d2e0fab4bbcc3762e70c95b30b48293c17d785d3db9877288a4efa75f03a749` |
| `fullmotion/detector/benchmarks/pads_p02_release_audit.json` | **Current PADS-P0.2 empirical evidence**: audit execution `PASS`, gate `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY`, 16/16 conditions; 13,447,168 samples stored exactly once in 10,318 single-row-group streams and all 13,447,168 read back, 14,729 segments over 4,411 detected gaps, 50,676 windows none crossing a segment, 23,928 bilateral window pairs with no sample-level alignment claim, five participant-disjoint folds, and byte-exact replay of all 10,318 streams read back from the store; evidence `7ca16981…` reproduced by two inode-disjoint processes | `8e5eb21cf8ecafcadc26a5a0bcdb37a4bd5bad0088a33bf42d1939b45b1f41eb` |
| `fullmotion/detector/benchmarks/pads_p01_dependency.json` | **Current P0.2 authority pin**: the exact P0.1 verdict, report bytes, source manifest and release counts P0.2 refuses to run without | generated from `FROZEN_DEPENDENCY`; equality asserted by `test_pads_p02_contract.py` |
| `fullmotion/detector/benchmarks/pads_p03_release_audit.json` | **Current PADS-P0.3 empirical evidence**: audit execution `PASS`, gate `PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION`, 16/16 conditions; 9,960 workload windows carrying 19,920 spectra on the frozen 3-12 Hz 37-bin grid, 6,077 independently audited windows across 862 strata, source and replay agreeing on every row, input hash and spectrum with a maximum bin error of exactly 0.0; evidence `a0be87d4…` reproduced by two inode-disjoint processes | `a2b6dfa3f598dfe7e2821285c3262dd1f817b168cb81e62683ad796445faf615` |
| `fullmotion/detector/benchmarks/pads_p02_dependency.json` | **Current P0.3 authority pin**: the exact P0.1 and P0.2.1 evidence hashes, published P0.2.1 report bytes, storage-index content hash, source manifest and both P0.2 schema fingerprints P0.3 refuses to run without | generated from `FROZEN_DEPENDENCY`; equality asserted by `test_pads_p03_contract.py` |
| `Tremora_IEEE_BigData_2026_research_brief.md` | **Superseded historical record**: abandoned participant-study and privileged-supervision scope; do not use for current claims or venue gates | `e28c563796816cb80febe1b9420b65230cc8de640e955ed98609865bda1d4c65` |
| `Tremora_Engineering_Calibration_Manifest_v0_2.xlsx` | **Frozen historical record**: not an execution gate for the public-data paper | `2aff25acc3025be7b04d67f6181deb443b878de423112d1175d45080b44f34db` |
| `Tremora_Dataset_Audit_Manifest.xlsx` | **Frozen historical record**: not an execution gate for the public-data paper | `3182d6813d4fbe18cf1606a28055b198f6314570746d0b5d9fccbb6ac9d97969` |

The superseded brief and both workbooks must remain byte-for-byte unchanged.
Any future status transition belongs in a new version of this registry or the
current specification, never as an in-place banner or formula change to those
historical files.

Implementation milestone: fullmotion commit
`b73d5ba3bffd3fb6ec815434263bc310de88f5f7` (`Add strict VIDIMU source parsers
and release audit`). This commit establishes source-format and release-inventory
evidence only; it does not establish video decoding, PTS association,
synchronization, a canonical VIDIMU snapshot, or comparative benchmark results.

Implementation milestone: fullmotion commit
`dd2df61ca68a30f5918a64d49ffad0edc72e2c70` (`Add PTS-preserving VIDIMU
decoder and exact CV pose-to-frame finalization`). This commit establishes the
generated-media Gate-A decoder, exact pose-to-frame association, atomic
success/source-failure outcome formats, and byte-identical replay across
distinct artifact files and roots. It is historical Gate-A evidence and does
not by itself attest that the replay tree came from an independent rerun.

Gate-B implementation milestones are fullmotion commits
`05667a43cb6fead809ee0b4bffd13cbfe9dd4cd5` (`Add trust-anchored VIDIMU
source snapshot materialization`),
`4503dec6fb3fe63b61222833fe5b4bfeaf7ed5b7` (`Freeze production
multi-detection CV estimator contract`), and
`12d806da2844047b35ff5a94f0189f02909904a7` (`Materialize trust-anchored
VIDIMU snapshot and freeze production CV finalization`). Gate B is `PASS`: all
208 videos were finalized twice from the same verified 624-asset snapshot in
separate processes and empty roots, producing 179,076 frame/result/selection
rows, 13,999 detections, no recorded failures, byte-identical per-record
Parquet, and canonical hash
`60c106a22416e424fad561d63bb5b4abe0e5eeef879856e11ddfcbfdbfa26a88`.
The bit-identity claim is limited to the two frozen observed macOS CPU
executions; execution receipts and inode disjointness are not cryptographic
remote attestation. This milestone does not establish video–IMU
synchronization, canonical clocks, frame–IMU indexes, multimodal windows,
spectra, storage/latency benchmarks, or paper-result claims, and it does not
by itself establish v0.5 synchronization authority.

The v0.5 authority audit is recorded by fullmotion commit
`1533aa9e54fa601ccef1f12c209534224e48ae9a` (`Audit VIDIMU clock authority and
fail closed`). The audit itself executed successfully and reproduced
byte-identically in two processes, but the empirical gate is
`NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`. The pinned sources do not provide an
authoritative RAW timestamp unit, RAW-to-video clock origin, or BodyTrack-row
to video-PTS contract. In addition, `S53_A13_T03` and `S57_A07_T01` contain
dual-direction applied overrides and are classified `AMBIGUOUS_SOURCE_MAPPING`.
No canonical clock tables, frame-to-IMU indexes, windows, spectra, or success
marker were produced. The three prescribed passing v0.5 commits were not
opened, and v0.6 remains closed pending new source authority or an explicitly
versioned change of study scope.

The separately versioned v0.5D audit is implemented by fullmotion commits
`06409da73f1c80200bee9bce64fbf2823f61051c` (`Add frozen VIDIMU
source-tool-output alignment authority contract`),
`5760b45db0f7502b7a55ac808cce35c53e35aeea` (`Reproduce VIDIMU RMSE-shift
and trimming transformations`), and
`658be1c8183254225124a6ab2803cdf63142fe69` (`Audit deterministic
source-derived VIDIMU alignment materialization`). Its source-procedure audit
passes: all 366 instructions and 217 non-MP4 derivatives are provenance-bound,
and every published CSV/MOT/RAW trim is exactly reproduced. Its materialization
gate is nevertheless `NO_GO`. The 208 RAW files contain 2,036,601 structural
five-sensor polling groups but only 299,711 source-authored dynamic 50 Hz
STO/MOT ordinals; the tools release supplies no mapping between those domains.
All 34 RAW trims also remove the N-pose rows and 30 end mid sensor group, while
the source modifier skips MP4. No `sto_alignment_contracts.parquet`,
`imu_tick_groups.parquet`, `derived_rate_contract.parquet`, validation table,
generic success marker, or `_STO_DERIVED_ALIGNMENT_SUCCESS` marker was
materialized. The requested final success commit was withheld, and v0.6D
remains closed pending a newly versioned, semantically corrected contract.


## Cross-dataset P0.1 branch

The cross-dataset timing-authority model, the Ego4D E4D-P0.1 machinery and the
PADS-P0.1 ingest audit are implemented on branch `e4d-p0.1-timing-authority`.
Eight frozen tiers decide in code what may be materialized; VIDIMU binds
`UNRESOLVED`, so excluding it from paired indexing is a type error rather than
a policy, and the v0.5/v0.5D reports stay hash-pinned by
`detector/tests/test_timing_authority_contract.py`.

**E4D-P0.1 is machinery only.** Ego4D requires a signed licence and pinned
assets that this project does not hold, so the audit has never been evaluated
against Ego4D data: the CLI returns `BLOCKED_INPUT_DATA_UNAVAILABLE` and exits
4. No Ego4D verdict, evidence hash, subset manifest, frame–IMU index or window
exists, and none may be cited.

**PADS-P0.1 is `PASS`.** The audit ran against the extracted PhysioNet 1.0.0
release and satisfied all fourteen conditions, including the two that bind the
whole release: structural reconciliation of 469 x 11 x 2 from the source
metadata, and independent reproduction across two child processes writing to
inode-disjoint empty output roots. Both conditions were part of the contract
before the first authoritative run, not added after seeing a result. As with
the v0.4 Gate-B and v0.5D receipts, the reproduction proof is execution
receipts under a trusted procedure in one frozen environment; it is not
cryptographic remote attestation. PADS-P0.2 — indexes, windows, spectra,
resampling ablations — remains closed.

**Provenance.** This branch is a re-derivation. An earlier build of the same
design was produced in an ephemeral environment and lost with it, together with
the adversarial reviews it had passed. The code here was rebuilt from the
architecture and has not carried those reviews; its commit messages say so.

**Correction.** An earlier pass recorded EgoInertia-MI as `UNVERIFIED_SOURCE`
and excluded it as probably fabricated. That was a search failure, not a fact.
arXiv:2607.03934 resolves to *EgoInertia-MI: A Multimodal Egocentric Vision and
IMU Benchmark for Motor Impairment Assessment* (Alhamdoosh, Pala, Mohamed,
Arvind), submitted 4 July 2026. The exclusion is withdrawn; the dataset remains
optional and off the critical path because its impairment is simulated by
healthy volunteers.


## PADS-P0.2 — authoritative indexing and gap-aware replay

PADS-P0.2 is `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY` on all sixteen conditions,
materialized twice in separate processes against the frozen PhysioNet 1.0.0
release. Every source sample is stored exactly once and every one is read back — the
headline reconciliation holds in all three terms — and every one of the 10,318
streams replays byte-exactly when read back from the store, so the
reconstructed file hashes to the source asset's own SHA-256 for the whole
corpus rather than a frozen subset.

Two findings from the real corpus. The release contains **4,411 time gaps**
above `min(100 ms, 3 x dt_ref)`, so 10,318 streams yield 14,729 segments: a
fixed slicer anchored on sample counts would have produced windows spanning a
discontinuity. And of 50,676 windows only 23,928 bilateral pairs exist, because
a gap on one wrist removes a window that still exists on the other; those
windows stay unpaired rather than acquiring an invented partner.

Source time is stored in exact **picoseconds**. Every `Time` token in the
release carries ten decimal places, so `0.0099029541` s is 9,902,954.1 ns —
not representable as an integer nanosecond count. The requested `_ns` fields
would have rounded away a digit the source actually wrote.

Bilateral retrieval is `SOURCE_PROTOCOL_PAIR` with
`cross_wrist_clock_alignment = UNRESOLVED` and sample-level fusion refused on
every published row. P0.2 emits no spectrum, tremor frequency, band power,
resampled signal, anti-aliasing output, classification, video association or
comparative benchmark result, and publishes a zero count for each. Its success
marker is `_PADS_P02_INDEX_SUCCESS`, never a generic `_SUCCESS`.

The reproduction proof is two child processes writing to inode-disjoint empty
output roots and agreeing on the evidence hash. As with P0.1, v0.4 Gate B and
v0.5D, that is execution receipts under a trusted procedure in one frozen
environment; it is not cryptographic remote attestation.

The materialized store itself — about 910 MB of Parquet per run — lives beside
the dataset and is not committed to this repository. PADS-P0.3, P0.4 and P0.5
remain closed.


## PADS-P0.3 — source-time spectral preservation

PADS-P0.3 is `PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION` on all sixteen
conditions, computed twice in separate processes against the frozen P0.2.1
store and the original release files.

The claim is narrow and it is a storage result: source-time, gap-aware storage
and indexed replay preserve the frequency-domain content of irregularly sampled
PADS wrist signals without interpolation, resampling or nominal-grid
substitution. It is not a disease classification, a tremor-detection accuracy,
a video–IMU or a rate-ablation result, and the report publishes a zero count
for every artifact those milestones would produce.

For 6,077 independently audited windows the original device files are re-parsed
by a second implementation that never calls the replay API, and the two paths
agree on every row, every time token, every value, every input hash and every
spectrum. The maximum observed bin error is exactly **0.0** — bit equality, not
a tolerance, because both paths feed identical float64 inputs into the identical
kernel. A future non-zero would be a reproducibility incident to diagnose, not
grounds to loosen the comparison.

Two probes are built to be discriminating rather than confirmatory. The
nominal-grid condition reports 0 substitutions *and* 9,960 windows whose stored
timestamps genuinely differ from an ordinal/rate grid, so the zero cannot be
vacuous. The Nyquist condition derives every limit from the stream's own
`dt_ref` and reports 0 rows carrying the declared 100 Hz rate's 50 Hz value —
possible here because no stream's `dt_ref` is exactly 10 ms.

The frequency grid is frozen at 3–12 Hz in 0.25 Hz steps, matching a
four-second window's Rayleigh resolution, with no zero-padding. Spectra come
from raw axes summed within a sensor family, never from vector magnitude: the
kernel controls demonstrate that a 5 Hz tone reports 5 Hz on its raw axis and
10 Hz through `|x|`. Those twelve controls run inside the audit process rather
than deferring to a test suite that may not have been executed.

The authoritative run pins single-threaded numerics and records that it did.
No BLAS call is made, so the result does not depend on threading.

The materialized spectra — about 32 MB per run — live beside the dataset and are
not committed to this repository. PADS-P0.4 and P0.5 remain closed.
