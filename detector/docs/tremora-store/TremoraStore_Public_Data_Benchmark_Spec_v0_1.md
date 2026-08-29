# TremoraStore Public-Data Benchmark Specification v0.1

**Status:** authoritative systems-paper scope; benchmark gates remain frozen  
**Date:** 2026-08-28  
**Provisional title:** **Tremora: A Timestamp-Native Storage and Replay Architecture for Video–IMU Motor Analytics**  
**Technical alternative:** **TremoraStore: Efficient, Gap-Aware Temporal Alignment of Video-Derived Motion and High-Rate IMU Streams**

## 1. Scope lock

This is a public-data systems paper.

- No new participant recruitment or recording.
- No clinical-validation claim.
- No claim that a new cohort establishes tremor accuracy or patient generalization.
- Tremor is the motivating healthcare analytics workload.
- The contribution is storage, indexing, synchronization, quality-aware window construction, and deterministic replay of heterogeneous temporal streams.

The central claim to test is:

> A timestamp-native, range-indexed representation can store each video-derived and inertial stream once, then deterministically reconstruct synchronized, gap-aware analysis windows with lower denormalization overhead or better access performance than appropriate baselines.

The comparative clause is empirical. The paper must report where Parquet loses as well as where it wins.

## 2. Superseded artifacts

The following artifacts remain immutable historical records but are not controls for this paper:

- Tremora_Engineering_Calibration_Manifest_v0_2.xlsx
- Tremora_Dataset_Audit_Manifest.xlsx
- Tremora_IEEE_BigData_2026_research_brief.md

Their byte hashes and supersession state are recorded separately in
`Tremora_Artifact_Status.md`; status notices must not be inserted into the
historical files themselves.

Do not populate EQA-001 or EQA-002. Do not revise their gates to fit this systems study. The reusable concepts are timestamp integrity, explicit offset/drift accounting, contiguous segmentation, immutable hashes, and versioned provenance—not participant recruitment, privileged-supervision inference, or clinical endpoints.

## 3. Snapshot model

Each published generation is immutable and complete:

~~~text
store/
├── CURRENT.json
└── snapshots/<snapshot_id>/
    ├── frame_index.parquet
    ├── cv_estimates.parquet
    ├── imu_samples.parquet
    ├── clock_map.parquet
    ├── frame_imu_index.parquet
    ├── window_index.parquet
    ├── window_rejections.parquet
    ├── provenance.json
    └── snapshot_manifest.json
~~~

The original video or immutable analysis proxy is retained once and referenced by hash. Frame extraction is not the default representation.

The snapshot manifest binds the generation ID, schema version, clock-map ID, window-policy ID, artifact paths, sizes, row counts, file hashes, semantic table hashes, schema fingerprints, canonical sort keys, and row-group policy. It also persists the complete alignment-pair plan and the complete continuity/window-generation policy, including all accepted and rejected continuity segments and every threshold needed to regenerate the derived indexes. Verification regenerates `frame_imu_index`, `window_index`, and `window_rejections` from the source tables and those plans, then requires exact agreement. The exact manifest payload, byte count, and digest accepted by that pass are carried as one binding into `CURRENT` resolution and verified replay pinning; the manifest is not independently reselected by pathname between those steps. An empty, missing, partial, or fabricated derived index therefore cannot pass merely because its rows are individually well formed.

Deterministic replay binds more than returned numeric arrays. Its semantic hash includes the logical window payload, any nearest-sample context outside the window-owned IMU range, the relevant clock-map rows, the provenance artifact digest, and semantic provenance such as stream conventions and generation policies. Physical Parquet layout, snapshot ID, and creation timestamp are excluded so logically identical snapshots remain comparable across row-group layouts; a unit, convention, policy, clock-quality, clock-map, or context-sample change must change the replay hash. Replay exposes provenance and clock-map metadata as immutable defensive copies.

On POSIX, publication synchronizes every staged file and directory transition before replacing `CURRENT.json`; a publisher-controlled fault may expose only the prior complete generation or the new complete generation. Windows retains file flushes and atomic `os.replace`, but v0.1 does not claim a POSIX-equivalent directory-entry fsync or crash-durability boundary there. Artifact creation uses exclusive no-follow handles and POSIX directory-relative opens. Python on Windows lacks the equivalent `dir_fd`/`openat` parent pin, so post-write inode checks fail closed on detected parent swaps, but filesystem access controls remain the concurrency boundary. The publisher creates and flushes the pointer payload, then performs its final artifact-and-manifest identity check immediately before pointer replacement. No userspace ordering can prevent independently writable snapshot files from being corrupted after that check, so release stores require filesystem access controls or immutable storage. Full verification rejects partial, aliased, mixed, unlisted, noncanonical, relationally inconsistent, or tampered generations. Verification streams each Parquet digest through a pinned descriptor and parses that same seekable descriptor; it does not materialize a second whole compressed-file byte buffer. Opening a verified replay session performs that full hash/semantic pass and then rehashes each artifact once while pinning live descriptors to close the verifier-to-pin race; each query thereafter uses predicate reads and bounded inode/size/time checks rather than whole-file rehashing. The session owns its descriptors and must be closed or used as a context manager. Warm replay may explicitly skip whole-snapshot hashes, but never path containment or canonical inventory checks. The snapshot-manifest hash recorded by `CURRENT.json` is an integrity binding within the store, not an external signature or proof of publisher authenticity; externally authenticated releases require a separately distributed signature or trusted digest.

Provenance records three distinct legal decisions: whether local analysis is allowed, whether source files may be redistributed, and whether derived artifacts may be released. Access status, license/terms hash, permitted-use statement, use decision, and release status must agree with those decisions. Local analytical permission never implies permission to redistribute source data or publish derived artifacts.

The v0.1 top-level provenance schema is closed. Unversioned extension fields—including clinical-validation or diagnostic-accuracy claim flags—are rejected instead of being preserved as unaudited assertions. A future extension requires an explicit schema version and validator.

## 4. Canonical temporal contract

### 4.1 Time and range semantics

- Native and canonical timestamps are signed integer nanoseconds.
- All query and ownership intervals are half-open: [start_ns, end_ns).
- Frame i owns [pts_i, pts_(i+1)); the final frame requires an explicit stream end.
- `effective_fps` describes that forward, frame-owned interval. The final frame uses [pts_last, stream_end); an earlier frame that terminates an epoch may use the preceding same-epoch interval when no forward same-epoch cadence exists. An isolated internal epoch frame with no derivation basis must store a null rate.
- `gap_before_ms` is a separate backward-looking diagnostic and must equal the time since the preceding frame.
- Exact end-boundary samples belong to the next interval.
- Ranges use a canonical start ordinal and exclusive stop ordinal.
- Empty ranges use start == stop; no sentinel is allowed.
- Nearest-sample ties select the earlier timestamp and then the lower canonical ordinal.
- Nearest-sample error is stored as signed integer nanoseconds; milliseconds are a reporting view.
- For non-interval-owned sample streams, the declared window/continuity domain supplies the cadence boundary when no outside neighbor exists. A first sample at least one nominal period after the start, or a final sample more than one nominal period before the half-open end, is an inferred missing edge sample: the window fails cadence and is rejected as a stream gap. Sub-period phase offsets at both edges remain valid, subject to the pinned rounding tolerance.
- For frame-owned streams, boundary coverage uses the first selected frame start and last fully owned interval end. A short excluded interval that straddles an accepted boundary exposes a missing boundary frame and rejects both accepted sides. If the first excluded straddler itself exceeds the configured video-gap threshold, the discontinuity belongs to the following half-open region and does not invalidate an otherwise clean left window.

### 4.2 Source and canonical order

Every frame and IMU sample retains:

~~~text
source identifier or sample index
source_ordinal
canonical_ordinal
native timestamp
canonical timestamp
stream ID
clock epoch ID
~~~

Source ordinal preserves acquisition order. Canonical ordinal is assigned after stable ordering by canonical timestamp and source ordinal. Logical ranges address canonical ordinals, never a possibly sparse source sample index.

### 4.3 Clock maps

Binary floating arithmetic on epoch-sized timestamps is prohibited. Each epoch uses an anchored rational map:

~~~text
canonical_ns = canonical_anchor_ns
             + round_nearest_even(
                   (native_ns - native_anchor_ns)
                   * scale_numerator / scale_denominator)
~~~

Each row contains:

~~~text
recording_id
stream_id
clock_epoch_id
continuity_component_id
acquisition_ordinal
source_start_ordinal
source_stop_ordinal
native_start_ns
native_end_ns
native_anchor_ns
canonical_anchor_ns
scale_numerator
scale_denominator
drift_ppm_derived
residual_p50_ms
residual_p95_ms
mapping_status
~~~

A timestamp reset starts a new clock epoch and a new continuity component. Native ranges may overlap across reset epochs; source-ordinal ranges may not. Consecutive epochs may share one continuity component only when their source-ordinal ranges, native clock domains, and rounded canonical domains are all adjacent within the pinned canonical tolerance. A native reset, native gap, or canonical forward gap requires a new component, and later acquisitions may never move backward in canonical time. `drift_ppm_derived` means `(canonical/native mapping scale - 1) × 10^6`; it is not the conventional fast-oscillator error sign. Mapping never extrapolates unless a separately versioned policy explicitly permits it.

`mapping_status = VALID` means only that the affine map is defined. Each alignment pair also pins `max_clock_residual_p95_ms`; an absent or excessive residual makes that mapped portion unusable for alignment even when its status is `VALID`. `UNRESOLVED` and `REJECTED` epochs, residual-policy failures, clock resets, and component discontinuities split usable continuity. Clock-component bounds may extend beyond the outer video domain so a valid pre-video predecessor or post-frame successor can participate in nearest-sample selection, but internal task/QC ledger boundaries remain hard cuts.

### 4.4 Stream multiplicity

Bare frame and sample indexes are insufficient. VIDIMU contains five IMUs and MMAct contains multiple video views. Therefore:

- frame and CV keys include video_stream_id;
- IMU keys include stream_id;
- range and window indexes include both video and IMU stream IDs;
- normalized index tables contain one row per frame/window and target-stream pair.

Creating a separate recording per camera view is not the default because it can duplicate shared IMU payloads.

All pairs sharing one video stream must declare the same `video_end_ns`, so the final frame has one owned interval regardless of target IMU. All continuity segments for one stored recording must share one immutable `split_group_id`; VIDIMU binds it to the source subject ID.

The generic Arrow schemas do not assert a universal coordinate system, body location, sensor unit, or channel interpretation. Provenance must provide an exact `recording_identity` mapping and a complete `stream_semantics` inventory for every planned video and IMU stream: source and stored keypoint/motion/orientation conventions, body location, payload kind, source and stored units, device/quaternion frames, canonicalization transform, and software version. An identity transform may be declared only when source and stored semantics are unchanged. Missing axes remain absent; adapters may not relabel quaternion orientation as raw acceleration or gyroscope.

## 5. Table contracts

### frame_index.parquet

One row per decoded frame, including stream/epoch keys, source and canonical ordinals, native PTS, canonical time, decode state, dimensions, effective cadence, gap metadata, and quality bits.

### cv_estimates.parquet

One row per CV output. Keypoints and other fixed-shape values use fixed-size numeric Arrow arrays, not JSON strings. Each row retains its frame key, canonical time, tracking quality, and estimator version. Replay never reruns the CV estimator.

### imu_samples.parquet

One row per source sample. Accelerometer, gyroscope, and quaternion channels are distinct nullable numeric columns. Missing axes remain null and are never inferred from orientation. A null in a channel required by the declared payload kind is permitted only with `INVALID_IMU_PAYLOAD`; that source row and ordinal remain in the table, intersecting windows are rejected, and valid surrounding windows remain available. Streams are stably ordered by recording, stream, canonical time, and source ordinal.

### clock_map.parquet

One row per piecewise clock epoch using the rational mapping contract.

### frame_imu_index.parquet

One row per frame and target IMU stream. It stores the frame-owned interval, stop-exclusive IMU ordinal range, nearest IMU ordinal, signed nearest-time delta, sample count, coverage, and alignment status. IMU values are not copied. If no common usable clock continuity exists, the complete inventory is represented explicitly by one `OUTSIDE_CONTINUITY` row per frame with an empty range, null nearest sample, and zero coverage; this state cannot yield valid windows.

If a frame's nearest IMU sample lies outside the frame/window-owned IMU range, replay returns it separately as nearest-sample context rather than silently widening the owned range. Its quality bits still participate in window rejection, and its semantic content is bound into the replay hash.

### window_index.parquet

One row per temporally valid window and required stream pair. It stores time bounds; frame and IMU ordinal ranges; counts; effective rates; coverage; cadence-regularity flags and deviation; qualified rate-based Nyquist ceilings; versioned quality/observability policies; three distinct frequency gates; and split group. `tremor_band_supported` records only temporal/rate/cadence/cycle support. `frequency_estimation_allowed` additionally requires valid CV tracking/keypoints and sufficient CV motion range. `valid_for_frequency` additionally requires sufficient IMU signal range. The corresponding CV-tracking, CV-motion-range, and IMU-signal-range diagnostics are persisted separately. These ceilings and gates are derived screening fields, not claims of a classical Nyquist limit under jitter or irregular sampling.

### window_rejections.parquet

One row per evaluated but rejected candidate, with the same identity/time bounds plus reason bits and stable reason codes. Robustness cannot be audited if invalid candidates simply disappear.

## 6. Public datasets and allowed roles

### VIDIMU — parsed source inventory; alignment remains unresolved

The [VIDIMU data descriptor](https://www.nature.com/articles/s41597-023-02554-9) reports 13 activities, video for 54 participants, simultaneous IMU for 16, 30 FPS video, and five 50 Hz sensors.

The pinned [VIDIMU v2.0.0 Zenodo release](https://zenodo.org/records/15075076) has a split extraction topology rather than co-located same-stem media. Paired pose/quaternion material is under `dataset/videoandimus/Sxx/Sxx_Axx_Txx.{csv,raw}`. Original-subtree camera candidates are under either `videosmallsize/videosoriginal/Sxx/Sxx_Axx_Txx.mp4` or `videosfullsize/videosoriginal/Sxx/Sxx_Axx_Txx.mp4`. The similarly named `videosbodytrack/*_pose.mp4` files contain rendered BodyTrack output and are optional QA artifacts, not camera input. A central-directory audit found same-stem original-subtree candidates for all 208 canonical CSV/RAW pairs in both video archives; only 206 pairs have same-stem BodyTrack QA renders. This inventory result does not validate media decoding, PTS, or image content. `VidimuCameraSourceAdapter` requires the exact archive wrapper and `videosoriginal` subtree, while the frozen v0.1 `VidimuAdapter` retains its historical BodyTrack-QA inventory meaning.

The strict source parser was exercised over all 208 canonical CSV/RAW pairs in the pinned release. It preserved all 10,184,045 RAW data rows one-for-one. The first five positional rows are N-pose calibration observations in the activity-specific five-sensor order; they are not identified by timestamp. Of the 208 records, 197 use zero N-pose timestamp tokens and 11 S41 records use nonzero large decimal tokens. Calibration and dynamic observations remain explicitly labeled source-row classes, not clock epochs. The payload is scalar-first WXYZ quaternion orientation, not released accelerometer or gyroscope axes. Twelve non-unit observations, all in `S54_A08_T02`, remain in the audit record and are represented with null quaternion channels plus `INVALID_IMU_PAYLOAD`; they are not normalized or dropped.

The deterministic checked release-audit manifest has SHA-256 `41277661f9e248da2f42c0703b69beec92bcaf0037b5d46264f64852ab22ecf1`. It binds the parser and audit implementation hashes, local `dataset.zip` hash, source-file hashes and per-record counts, and exact central-directory evidence for the two remote video archives. Replaying the audit over the pinned inputs produced byte-identical output. The published video-archive MD5 values were not recomputed; central-directory inspection does not hash member payloads or validate decodability.

The corresponding implementation milestone is fullmotion commit `b73d5ba3bffd3fb6ec815434263bc310de88f5f7` (`Add strict VIDIMU source parsers and release audit`).

The paper documents nominal 50 Hz sensor output, but the RAW format does not document how rows or timestamp tokens map to those hardware updates and contains repeated held values. Across the pinned paired subset, the per-stream held-observation fraction ranges from 0.797315 to 0.920376, with median 0.855143. RAW row cadence is therefore not promoted to sensor information bandwidth, and held observations are not deduplicated. Because the release does not document the timestamp unit, clock source, or relation to video PTS, the parser preserves the exact decimal token and uses context-independent integer string parsing only for equality and ordering checks. It emits no `sensor_time_native_ns`, `canonical_time_ns`, canonical `imu_samples`, or `clock_map` rows. Clock status remains unresolved and canonical materialization remains deferred.

The paired BodyTrack CSV has 34 XYZ positions in millimetres and no timestamp,
frame identifier, confidence, or visibility column. The parser preserves row
ordinal and an exact-zero-triplet sentinel without inferring tracking validity.
The pinned audit found 267 fully zero rows across 13 files. Those BodyTrack rows
remain unbound to video. The v0.3 PTS/CV contract instead binds newly run CV
outputs directly to decoder-owned frame identities; it does not associate the
released BodyTrack rows by count or row position.

Published synchronization operates on derived 50 Hz products and estimates an
integer shift; it does not establish an absolute RAW-to-video clock residual or
a valid affine clock map. Filename co-occurrence establishes inventory only.
The generated-media Gate-A contract covers PTS-preserving decode, explicit
orientation, decoder-owned frame identities, one-result-per-frame CV binding,
atomic artifacts, and exact replay. Gate B has now materialized the pinned
public release as a content-addressed source snapshot: three source objects and
624 extracted assets resolve exactly across all 208 inventory records, with no
unavailable, ambiguous, or unreferenced assets. The production estimator binds
the model manifest, weights, preprocessing configuration, runtime lock, and
association contract while preserving zero-, one-, and multi-detection frame
outcomes and keeping primary-hand selection separate.

Two clean source-to-CV executions ran in separate processes and empty roots.
Their 208 seven-file recording bundles contain 179,076 decoded frames, 179,076
CV frame results, 179,076 selection rows, and 13,999 detections; all decode,
timing, inference, foreign-key, and selection failure counts are zero. Every
per-record Parquet artifact is byte-identical across the two roots, and both
runs have canonical content hash
`60c106a22416e424fad561d63bb5b4abe0e5eeef879856e11ddfcbfdbfa26a88`.
The release audit returns `BYTE_IDENTICAL_SOURCE_TO_CV_PASS` / `PASS` and has
SHA-256 `d24863d20347cf2c9ab092a9f7771ada3a88ec8fbc77a7b33788df5c0637a10e`.
Execution receipts and disjoint root/artifact inodes establish the two local
executions under the trusted procedure; they are not cryptographic remote
attestation. Because the frozen manifest honestly records
`deterministic_mode=false` and an unconstrained thread count, bit identity is
claimed only for these two executions in the frozen observed macOS CPU
environment, not across arbitrary platforms or runtimes.

Synchronization, accepted continuity, frame-to-IMU ranges, canonical clocks,
frequency-eligible windows, spectra, and comparative benchmarks remain
unopened. VIDIMU may not serve as independent truth for raw-axis tremor spectra,
drift estimation, or piecewise clock-map accuracy.

The corresponding Gate-A implementation milestone is fullmotion commit
`dd2df61ca68a30f5918a64d49ffad0edc72e2c70` (`Add PTS-preserving VIDIMU
decoder and exact CV pose-to-frame finalization`). Gate B is implemented by the
bounded fullmotion commits `05667a43cb6fead809ee0b4bffd13cbfe9dd4cd5`
(`Add trust-anchored VIDIMU source snapshot materialization`),
`4503dec6fb3fe63b61222833fe5b4bfeaf7ed5b7` (`Freeze production
multi-detection CV estimator contract`), and
`12d806da2844047b35ff5a94f0189f02909904a7` (`Materialize trust-anchored
VIDIMU snapshot and freeze production CV finalization`). Gate B is `PASS`; this
did not itself establish synchronization authority.

#### VIDIMU v0.5 RAW-native synchronization-authority audit — `NO-GO`

The v0.5 evidence audit executed successfully, but the RAW-native clock gate
returned `NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`. This is an empirical authority
failure, not an audit-runtime failure. The audit binds the pinned article,
Zenodo record metadata, complete 15-file VIDIMU-TOOLS v1.0 release, native
source parser and v2 release audit, v0.4 snapshot and frame evidence, analysis
archive, and dataset synchronization subtree by content hash.

An independent scan verified all 208 original RAW assets: 545,308,276 bytes,
10,184,045 rows including 1,040 N-pose rows, 10,183,005 dynamic observations,
and 1,040 sensor streams. Source timestamp tokens are strictly increasing
within every stream, with zero duplicate or reversing tokens. The scan found
8,735,242 exact consecutive held payload rows; per-stream held fractions range
from 0.797315 to 0.920376, with median 0.855143. These are unitless source-token
and payload observations, not a claim that RAW row cadence is sensor
information bandwidth.

The gate remains closed for five reasons:

- the audited pinned authority does not define the RAW decimal timestamp unit;
- it does not bind the RAW clock origin to recording start or video PTS;
- equal BodyTrack CSV-row and decoded-frame counts for all 208 recordings do
  not establish an authoritative row-to-PTS mapping;
- the nominal 50 Hz acquisition statement does not authorize treating every
  RAW row as one native hardware sample or supply a trusted selection rule; and
- two records have conflicting applied source transformations.

The two ambiguous records are `S53_A13_T03`—plot-selected VIDEO cut 2 plus an
applied IMU cut 1—and `S57_A07_T01`—plot-selected VIDEO cut 11 plus an applied
IMU cut 14. Their released CSV, MOT, and RAW derivatives preserve both
directional paths, so both are classified `AMBIGUOUS_SOURCE_MAPPING` rather
than silently assigned to the plotted winner. Separately, all 34 synchronized
RAW derivatives remove the five N-pose rows, and 30 begin mid five-sensor
cycle. That defect makes those derivatives unusable as strict native-sample
tables; it does not imply that the 208 original RAW files are malformed.

No canonical frame or IMU clock table, synchronization map, validation signal,
frame-to-IMU index, window, spectrum, or `_SUCCESS` artifact was emitted. v0.6
therefore remains closed. The byte-identical checked authority report has
SHA-256 `3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec`
and is recorded by fullmotion commit
`1533aa9e54fa601ccef1f12c209534224e48ae9a` (`Audit VIDIMU clock authority and
fail closed`).

Advancing requires new authority, not a formula change: either obtain a
source-authoritative RAW unit/origin, BodyTrack-row-to-video-PTS contract, and
resolution of the two dual-direction cuts; select another paired dataset with
documented clocks; or approve a separately versioned, narrower STO-derived
orientation-clock study whose investigator-defined assumptions are stated
explicitly. The source-authored 50 Hz STO timeline may not be silently
substituted for native RAW timestamp authority.

#### VIDIMU v0.5D source-tool-output derived-alignment audit — `NO-GO`

v0.5D was opened as a separately versioned attempt to preserve and replay the
published VIDIMU shift-and-trim procedure without reopening the v0.5
RAW-native clock decision. Its authority enum distinguishes raw shared clocks,
raw mapped clocks, source-derived alignment, heuristic alignment, ambiguous
source alignment, and unresolved alignment. The v0.5 status remains permanently
`NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`.

The source-procedure portion passed. All 366 source-order `infoToSync.csv` rows
were exact-byte bound, including 217 non-MP4 instructions spanning 181
recordings. Replaying the published formulas—CSV removes `CutFrames`, MOT
removes `floor(CutFrames × 5/3)`, and RAW removes
`floor(CutFrames × 25/3)`—reproduced all 217 released derivatives
byte-for-byte while preserving source line endings. The two dual-direction
records, `S53_A13_T03` and `S57_A07_T01`, remain
`AMBIGUOUS_SOURCE_ALIGNMENT` and receive no chosen mapping, range index,
window, or eligibility override.

The complete materialization nevertheless returned `NO_GO`. Across all 208
originals, the RAW files contain 2,036,601 complete structural five-sensor
polling groups, whereas the source-authored 50 Hz STO and MOT products contain
299,711 dynamic ordinal rows. The aggregate ratio is 6.795216, with
per-record ratios from 5.261773 to 7.865480. The released tools contain no
RAW-group-to-STO/MOT selection map, so those RAW groups cannot honestly be
called 50 Hz IMU ticks. In addition, every one of the 34 RAW trims removes the
five N-pose rows and 30 ends mid five-sensor group. Exact source replay is
therefore incompatible with the proposed post-trim `imu_tick_groups` contract.
The source modifier also skips MP4, so promoting its CSV cut to decoded
video-frame ordinals would add a mapping that the published transformation did
not perform.

The checked audit accordingly records audit execution `PASS` but
`source_derived_alignment_gate = NO_GO`, with 217 overrides expected, bound,
and reproduced; zero unreproduced overrides; zero eligible materialized pairs;
zero canonical clocks or clock segments; and zero alignment Parquet files,
generic success markers, or `_STO_DERIVED_ALIGNMENT_SUCCESS` markers. Its
SHA-256 is
`131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0`.

The bounded implementation commits are
`06409da73f1c80200bee9bce64fbf2823f61051c` (`Add frozen VIDIMU
source-tool-output alignment authority contract`),
`5760b45db0f7502b7a55ac808cce35c53e35aeea` (`Reproduce VIDIMU RMSE-shift
and trimming transformations`), and
`658be1c8183254225124a6ab2803cdf63142fe69` (`Audit deterministic
source-derived VIDIMU alignment materialization`). The requested final success
commit and v0.6D were not opened because no alignment artifact was
materialized.

A corrected future contract must keep RAW polling groups as provenance-only
evidence, define nominal 50 Hz ordinals from source-authored STO/MOT rows,
explicitly bind MOT-to-STO ordinal identity, represent 208 pair decisions
separately from the 366 source instructions and 217 trim overlays, and state
any CSV-to-decoded-frame positional assumption. That correction requires a
new version; it may not alter the v0.5D `NO-GO` report.

### MMAct — scale and multi-view workload

The [official MMAct page](https://mmact19.github.io/2019/) reports seven modalities, more than 1,900 untrimmed 1080p/30 FPS videos, more than 40,000 instances, 40 participants, and four surveillance views plus an egocentric view. Access is request-based and its academic-use/non-redistribution terms must be respected.

Use it for storage volume, ingest throughput, index construction, multi-view/shared-IMU handling, random access, and batch loading. It is not clinical tremor data.

### Ego4D IMU — real irregularity stress input

The [Ego4D IMU documentation](https://ego4d-data.org/docs/data/imu/) describes normalized canonical-video timestamps and documents null canonical timestamps, partial component coverage, missing acceleration, extreme timestamps, non-monotonic records, and absent IMU calibration.

Use it for deterministic detection, segmentation, stable ordering where unambiguous, and rejection where ambiguity remains. Its canonical mapping assumes initial alignment to container origin, so Ego4D is not independent clock ground truth.

### PADS — tremor-aware IMU-only workload

The [PADS PhysioNet release](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) contains 5,159 assessment steps from 469 individuals with bilateral 100 Hz acceleration and gyroscope. It has no paired camera stream.

Use it only for raw-axis schema coverage, tremor-band metadata, spectral round-trip preservation, and high-rate window-query performance. Do not use it for video–IMU alignment claims.

**PADS-P0.1 ingest is `PASS`.** All fourteen conditions are satisfied against the real release: 469/469 participants, 5,159/5,159 assessment steps and 10,318/10,318 device files reconcile from the source metadata; 11,256 referenced files verify against the release's own `SHA256SUMS.txt`; 13,447,168 samples parse with every one of their 80,683,008 sensor values usable, no duplicate or non-monotonic Time, and no cadence or span deviation from the declared 100 Hz. Two separate processes writing to inode-disjoint empty roots produced the identical evidence hash `e25ce02f…`. See [`pads_p01_release_audit.json`](../../benchmarks/pads_p01_release_audit.json).

The timing contract is `SOURCE_RELATIVE_UNIMODAL_CLOCK` with `relative_time_basis = SOURCE_TIME_COLUMN`: the release publishes a per-sample `Time` channel and the declared rate validates that timeline rather than generating it. The published clock is a real device clock — intervals in the first file run from 7.13 ms to 12.90 ms around a 9.99 ms median — so only the median interval is compared against the declared period. No window, spectral feature or video association is produced, and PADS-P0.2 stays closed until it is opened as its own milestone.

### Optional datasets

EgoInertia-MI is a verified release: [arXiv:2607.03934](https://arxiv.org/abs/2607.03934), *EgoInertia-MI: A Multimodal Egocentric Vision and IMU Benchmark for Motor Impairment Assessment* (Alhamdoosh, Pala, Mohamed, Arvind), submitted 4 July 2026, with synchronized egocentric video and wearable IMU over 19 activities and three simulated severity levels. An earlier pass in this project recorded it as unverifiable and excluded it; that was a search failure, not a fact, and the exclusion is withdrawn. Simulated impairment in healthy volunteers is still not clinical validation, so it remains optional, off the critical path, and subject to the same audit path as any other dataset.

## 7. Research questions and measurements

### RQ1 — Storage efficiency

Measure per representation and recording hour:

~~~text
total bytes
payload bytes versus index/provenance overhead
compression ratio against the same canonical payload
physical sample rows
sample references
duplicate payload values
file count
~~~

### RQ2 — Retrieval efficiency

Use the same machines, recordings, windows, and returned columns:

~~~text
cold and warm single random 4-second latency
p50/p95 over 10,000 seeded requests
batch-of-64 latency
sequential replay samples/second and recording-hours/second
peak resident memory
CPU time and utilization
~~~

Report storage medium, cache procedure, thread count, row-group/chunk size, compression level, and library versions.

### RQ3 — Temporal correctness

The range index must match independent brute-force selection:

~~~text
lo = lower_bound(canonical_times, start_ns)
hi = lower_bound(canonical_times, end_ns)
~~~

Measure range-boundary error, signed nearest-sample error, count agreement, coverage, materialized clock-map agreement, and deterministic semantic replay hash. Synthetic hidden canonical clocks validate mapping accuracy; an index agreeing with itself is not ground truth.

### RQ4 — Failure robustness

Measure detection, segmentation, false acceptance, false rejection, valid-data retention before and after faults, and deterministic reason codes.

### RQ5 — Spectral preservation

Compare an in-memory canonical stream with replay of that same mapped stream:

~~~text
dominant frequency
spectral centroid
band power
PSD correlation
cross-modal lag
numeric payload bit preservation
~~~

Clock correction and storage losslessness are separate experiments. Correcting drift may intentionally change frequency scale; correcting offset may intentionally change cross-modal lag.

## 8. Baselines

| ID | Representation | Fair interpretation |
|---|---|---|
| B0 | Raw CSV plus runtime timestamp join | No payload duplication; repeated parsing and alignment. |
| B1a | Per-frame JSON with disjoint frame-interval IMU payload | Does not inherently duplicate samples; measures serialization, update isolation, and metadata overhead. |
| B1b | Overlapping-window JSON with copied IMU payload | Denormalized training format; duplicates samples across overlapping windows. |
| B2 | Monolithic HDF5 with equivalent compression, provenance, and query semantics | Strong array baseline; HDF5 is not presumed weaker. |
| M1 | Parquet streams, rational clock map, range indexes, and immutable snapshot manifest | Proposed method. |

The paper may claim a duplication advantage only against a baseline that actually duplicates. HDF5 receives the same metadata and query opportunity as M1.

The current synthetic harness is only a matched-core writer and duplication-accounting smoke test. It gives all writers the same three-table logical payload and byte-identical provenance, but M1 is not yet a complete seven-table snapshot and the harness does not run retrieval queries. The harness rejects durations of four seconds or less so its B1b label always denotes genuinely overlapping windows. Its per-writer times and bytes are diagnostics only: no cross-format ranking, ratio, or paper claim is permitted. The empirical benchmark begins only after all baselines implement the complete canonical payload, legal/provenance material, equivalent compression policy, and matched cold/warm queries.

## 9. Controlled anomaly matrix

Derived copies come from immutable public sources. A corruption manifest records source hash, seed, transformation order, parameters, code version, and output hash.

~~~text
clock offset: ±50, ±100, ±250, ±500 ms
clock drift: ±50, ±100, ±500 ppm
random IMU loss: 1%, 5%, 10%
contiguous IMU burst loss: matched durations/rates
random frame loss: 1%, 5%, 10%
contiguous frame burst loss: matched durations/rates
one timestamp reset
one reversed source-ordinal block
one duplicated timestamp block
partial modality coverage
variable video cadence
variable IMU cadence
~~~

Expected behavior:

- known offset/drift maps to hidden canonical time within prespecified tolerance;
- a reset creates a new epoch;
- gaps and unresolved boundaries reject crossing windows;
- valid windows before and after a fault remain accessible;
- stable reordering occurs only when source order resolves ambiguity;
- ambiguous rows remain in the anomaly ledger and out of valid indexes;
- source files are never modified.

Random and burst loss are separate conditions because equal percentages need not create equal gaps.

## 10. Tremor-aware observability policy

The initial policy may compute:

~~~text
video_rate_based_nyquist_hz = 0.5 × effective_video_fps
imu_rate_based_nyquist_hz = 0.5 × effective_imu_hz
video_observable_max_hz = min(12, 0.4 × effective_video_fps)
imu_observable_max_hz = 0.5 × effective_imu_hz
~~~

The 0.4 multiplier is a conservative versioned video-observability policy, not a universal physical law. The field names deliberately say *rate-based Nyquist*: under tolerated jitter or irregular sampling, neither field is a classical sampling-theorem guarantee. `tremor_band_supported` requires the requested upper band to be strictly below both qualified observability limits; exact rate/2 is unsupported. It also depends on pinned cadence, maximum-gap, coverage, window-length, and minimum-cycle policies. A 15 FPS window therefore has a rate-based ceiling of 7.5 Hz but a more conservative provisional video-observability limit of 6 Hz; satisfying either screening value alone does not establish observability.

The next gate, `frequency_estimation_allowed`, requires `tremor_band_supported` plus valid tracking/keypoint coverage and a versioned minimum CV-motion peak-to-peak range expressed in that stream's declared stored units. The final `valid_for_frequency` gate additionally requires a versioned, payload-aware IMU signal policy: stored-unit peak-to-peak thresholds for acceleration and angular velocity, a sign-invariant quaternion orientation-set peak-to-peak geodesic range, and minimum counts of varying CV components and IMU channels. Quaternion rows must be finite and unit-normalized within the pinned tolerance; `q` and `−q` represent the same orientation and cannot create false motion. Exact floating-point jitter must not satisfy a physical signal-range gate.

For irregular sampling, rate/2 is not an unconditional Nyquist limit. The rate-based ceiling is nullable or disqualified unless cadence and gap policy pass.

## 11. Implementation order

1. Freeze Arrow schemas and interval/range semantics.
2. Implement rational clock epochs and timestamp verification.
3. Implement normalized multi-stream frame-to-IMU indexes.
4. Implement valid-window and rejection-ledger generation.
5. Implement atomic snapshot writing, hashes, and deterministic replay.
6. Add a conservative VIDIMU inventory adapter.
7. Add a PTS-preserving offline CV-to-Parquet runner for the existing estimator.
8. Materialize the first VIDIMU snapshot.
9. Add MMAct and Ego4D adapters.
10. Implement B0, B1a, B1b, and B2 with matched semantics.
11. Run anomaly, storage, latency, correctness, and spectral experiments.
12. Generate tables and figures from immutable result manifests.

## 12. Systems-paper go/no-go gate

The paper advances only if all conditions hold:

1. Every public source and derived corruption has an immutable hash and a legal-use record that separately resolves local analysis, source redistribution, and derived-artifact release.
2. Every materialized canonical timestamp exactly recomputes from its pinned clock map.
3. Every sample has unambiguous stream, epoch, source-order, and canonical-order identity.
4. Each source IMU payload is stored once in M1.
5. Random and sequential replay are deterministic at the logical-content level.
6. Gap, reset, task, CV-validity, and unresolved-map boundaries never enter valid windows.
7. Anomaly outcomes match the corruption manifest while preserving valid surrounding data.
8. Canonical in-memory and replayed spectra agree within prespecified tolerances.
9. M1 materially improves at least one primary systems outcome over B0/B1/B2 without an unacceptable regression elsewhere.
10. The manuscript makes no clinical-accuracy, diagnostic, or patient-generalization claim.

Failure is reported honestly. Baseline definitions, compression settings, and gates are not changed after viewing results merely to produce a win.

## 13. Venue status

The [IEEE BigData 2026 Healthcare Data call](https://bigdataieee.org/BigData2026/calls/special-healthcare/) includes healthcare-data acquisition, storage, processing, information systems, mobile solutions, and wearable health information. Its posted full-paper deadline is **August 29, 2026**. No new trial is required by scope; readiness depends on completing the public-data benchmark and making the healthcare-data relevance concrete.

The [Machine Learning on Big Data call](https://bigdataieee.org/BigData2026/calls/special-machine-learning/) includes architectures, systems, temporal/spatiotemporal/streaming data, heterogeneous sources, and practical implementation and evaluation. Its posted full-paper deadline is **September 30, 2026**.

Current status:

| Component | Status |
|---|---|
| Paper scope and storage contract | Frozen v0.1; historical artifacts separately hashed |
| Rational clock/range/window core | Implemented; adversarial synthetic regression tests only |
| Durable snapshot and deterministic replay | Implemented; adversarial synthetic regression tests only |
| Matched-core storage harness | Implemented as a smoke test; cross-format claims prohibited |
| VIDIMU trust-anchored source snapshot v0.4 | `PASS`; three source objects and 624 assets hash-verified and exactly reconciled across all 208 records |
| VIDIMU production CV finalization v0.4 | Gate B `PASS`; two clean, process/root/inode-disjoint all-208 executions produced byte-identical per-record Parquet and canonical hash `60c106a2…`; proof is bounded to the frozen observed environment |
| VIDIMU synchronization and canonical materialization | Unopened; raw-to-video clocks, accepted continuity, frame–IMU indexes, windows, and spectra remain unresolved |
| Cross-dataset timing-authority model | Implemented and frozen; VIDIMU binds `UNRESOLVED`, Ego4D `SOURCE_CANONICAL_TIMESTAMP`, PADS `SOURCE_RELATIVE_UNIMODAL_CLOCK` |
| E4D-P0.1 Ego4D timing-authority audit | Machinery implemented; empirical gate not evaluated — the CLI returns `BLOCKED_INPUT_DATA_UNAVAILABLE` pending the signed licence and pinned assets |
| PADS-P0.1 ingest audit | `PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK`; 14/14 conditions on the real release, reproduced across two processes |
| MMAct scale benchmark | Pending access and ingestion |
| Ego4D irregularity benchmark | Pending ingestion |
| PADS storage/spectral benchmark | Closed pending its own milestone; P0.1 emits no window or spectrum |
| Comparative systems results | None yet |
| BigData Healthcare | Conditional on benchmark completion |
| MLBD | Strong backup; conditional on benchmark completion |
