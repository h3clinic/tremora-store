# TremoraStore v0.1

This package is the public-data systems layer for timestamp-native video–IMU
storage and deterministic replay. It does not collect participant data, train a
privileged camera model, or make a clinical-validation claim.

## Snapshot layout

```text
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
```

Original video stays outside the exact nine-file snapshot inventory and is
referenced and hashed in provenance. The core neither accepts extra proxy/link
entries nor extracts millions of image files by default.

## Locked semantics

- All analysis intervals are half-open: `[start_ns, end_ns)`.
- Frame `i` owns `[pts_i, pts_(i+1))`; the caller supplies the final frame end.
  Per-frame `effective_fps` is therefore derived from the next timestamp in the
  same clock epoch. An internal epoch's final frame may fall back to its
  same-epoch predecessor; the stream's final frame uses its explicit tail
  `[pts_last, video_end_ns)`. An isolated frame with no same-epoch neighbor and
  no final-stream tail basis must store a null `effective_fps` rather than an
  invented cadence. `gap_before_ms` remains explicitly backward-looking.
- Range stops are exclusive canonical ordinals; an empty range has equal start
  and stop values.
- Exact end-boundary samples belong to the next range.
- Nearest-sample ties choose the earlier timestamp, then lower ordinal. The
  stored delta is signed integer nanoseconds.
- Native-to-canonical clock maps use anchored integer ratios and nearest-even
  rounding. Float multiplication of epoch-sized timestamps is prohibited.
- Clock-map `VALID` means that the rational map is defined, not that cross-modal
  synchronization quality is acceptable. Every alignment pair pins a maximum
  p95 residual. Missing, excessive, `UNRESOLVED`, and `REJECTED` clock domains
  split usable continuity and cannot be covered by accepted segments.
- Each timestamp reset receives a new `clock_epoch_id`. Native timestamp ranges
  may overlap across epochs, but source-ordinal ranges may not. Two epochs may
  share a continuity component only when their source ordinals, native clock
  domains, and rounded canonical domains are all adjacent; a native reset or
  gap therefore forces a new component.
- Required IMU channels may remain null only when the source row carries
  `INVALID_IMU_PAYLOAD`. The row and source ordinal are retained for anomaly
  provenance, every intersecting window is rejected, and valid surrounding
  windows remain available.
- Window creation emits a valid temporal index and a rejected-candidate ledger.
- Window frame ranges include only frames whose complete owned interval lies
  inside the candidate. Alignment rows crossing a declared continuity boundary
  cannot enter a valid window.
- Gap detection inspects predecessor/successor cadence at window boundaries;
  an observed discontinuity cannot disappear merely because it straddles two
  candidates. Frame-owned coverage uses the first frame start and final fully
  owned interval end. A short excluded interval straddling an accepted boundary
  exposes a missing boundary frame and rejects both sides; an excluded interval
  that itself exceeds the configured gap threshold assigns that discontinuity
  to the following half-open region and does not poison the clean left window.
  When no outside neighbor exists, the declared temporal domain is the boundary:
  a first IMU sample at least one nominal period after the start, or a final
  sample more than one nominal period before the half-open end, fails cadence
  and is rejected as a stream gap. Sub-period phase offsets at both edges remain
  valid.
- `video_rate_based_nyquist_hz` and `imu_rate_based_nyquist_hz` are rate-based
  Nyquist ceilings (half the observed count-per-window rate), not claims of a
  classical Nyquist guarantee under jitter. They are null unless cadence passes
  the versioned regularity gate. The stricter video observability ceiling uses
  `min(cap, 0.4 × effective FPS)`. Gaps, cadence irregularity, coverage, window
  duration, and minimum cycle count remain separate gates, and an exact ceiling
  equality is never accepted.
- `tremor_band_supported` is only the cadence/rate/cycle screen.
  `frequency_estimation_allowed` additionally requires the pinned tracking,
  valid-keypoint, and camera-motion range gates. `valid_for_frequency` further
  requires the paired IMU range gate. Motion/IMU thresholds are persisted per
  stream pair in stored units; exact inequality or one-ULP jitter is not signal
  evidence. Quaternion variation uses sign-invariant angular distance because
  `q` and `-q` represent the same orientation, and quaternion payloads must be
  unit-normalized within an absolute norm tolerance of `1e-3`. These are
  storage-layer eligibility screens, not SNR, clinical validity, or proof that
  a frequency estimate is accurate.
- Spectral losslessness compares an in-memory canonical stream with replay of
  the same mapped stream. Clock correction itself may intentionally change
  frequency scale or cross-modal lag.
- Snapshot publication fsyncs files and directory transitions before replacing
  `CURRENT.json`. The pointer's snapshot-manifest digest binds that pointer to
  the selected manifest within the store's integrity model. It is not a digital
  signature, MAC, transparency-log proof, or other source of external
  authenticity. Directory-entry fsync, and therefore the strongest stated
  crash-durability boundary, is POSIX-only in v0.1. On Windows, files are still
  flushed and `os.replace` remains atomic, but the package does not claim a
  POSIX-equivalent directory-flush guarantee. The publisher performs a final
  artifact-and-manifest identity check immediately before pointer replacement;
  no userspace ordering can prevent a separately writable artifact from being
  corrupted after that check, so release stores still require filesystem access
  controls or immutable storage. Artifact creation uses exclusive, no-follow
  handles and POSIX directory-relative opens. Windows lacks the equivalent
  `dir_fd`/`openat` parent pin in Python's standard library, so post-write inode
  checks fail closed on detected parent swaps, but access controls remain the
  concurrency boundary.
- Full verification streams each Parquet digest through a pinned descriptor and
  parses that same seekable descriptor, avoiding a whole-compressed-file Python
  byte buffer. It checks file and semantic hashes, canonical schemas,
  legal-use provenance, clock recomputation, foreign keys, and exact generated
  alignment/window completeness. The exact manifest payload, size, and digest
  validated in that pass remain one binding through `CURRENT` resolution and
  replay pinning. Integrity-verified replay (`verify=True`) and
  explicitly unchecked warm replay (`verify=False`) are different benchmark
  protocols. Warm replay may skip hashes and relational regeneration, but never
  canonical inventory or path containment; its latency must not be reported as
  verified replay latency.
- Opening a verified replay session first performs the full verifier's hash and
  semantic pass, then rehashes each artifact once while pinning live descriptors
  to close the verifier-to-pin race. Each random query uses Parquet predicates
  and bounded inode/size/time checks before and after reading; it does not rehash
  or copy whole artifacts per window. `RecordingStore` owns those descriptors;
  callers should use it as a context manager or call `close()`. Deterministic
  replay exposes immutable provenance, the applicable
  clock map, and `imu_nearest_context` for nearest samples outside the
  window-owned IMU range, then binds their semantic content into the replay
  hash. Quality bits from that nearest-sample context also participate in
  window rejection, so an invalid halo sample cannot support a nominally valid
  frame alignment. Re-encoding Parquet row groups or changing a snapshot ID
  does not change that logical hash; changing units, frames, transforms, clock
  quality, or nearest-sample context does.

Every ingestion requires a versioned legal-use record with a dataset/version,
record URI, source hashes, license/terms hash, processing commit, estimator
version, and observability policy. Local analysis, source redistribution, and
derived-artifact release are separate boolean gates; their access, decision,
redistribution, and release statuses must agree. Permission to analyze locally
does not imply permission to redistribute source bytes or publish derivatives.
The v0.1 top-level provenance contract is closed: unversioned extension fields,
including clinical or diagnostic claim flags, are rejected rather than stored
as unaudited assertions.

Generic Arrow columns intentionally carry no invented universal IMU units,
device frame, quaternion convention, CV coordinate convention, or hand-scale
unit. Provenance must exactly cover every stored stream with source semantics,
stored semantics, body location, payload kind, and a versioned canonicalization
transform (including an explicit identity transform). The snapshot also binds
its internal `recording_id` to the public source recording ID.

Provenance also persists the complete generation recipe: every video/IMU pair
and final-video boundary used to regenerate `frame_imu_index`, plus the full
window policy and all accepted or rejected continuity segments used to
regenerate `window_index` and `window_rejections`. This makes an empty or
partially deleted derived table distinguishable from a genuinely empty result.
Every pair sharing a video stream must use the same final-video boundary, and
every continuity segment for one recording must use one immutable
`split_group_id`. VIDIMU additionally binds that group to its source subject ID.

The current `benchmark_storage.py` is a synthetic matched-core writer and
duplication-accounting smoke harness. Its M1 payload is not a complete snapshot,
so its bytes and writer times are not cross-format performance evidence. It
records verified and unchecked/warm replay as separate, unmeasured future
protocols and requires more than four seconds of input so B1b actually contains
overlapping windows; public-data retrieval results begin only with matched
complete representations and equivalent queries.

VIDIMU v2.0.0 now has two deliberately separate adapter roles. The frozen
`VidimuAdapter` remains the v0.1 metadata/provenance inventory for
`videosbodytrack/*_pose.mp4`; those videos contain rendered BodyTrack output and
are QA artifacts, not camera inputs. `VidimuCameraSourceAdapter` is the only
future CV-ingestion entry point. It requires the official archive wrapper and
both official `videosoriginal` and `videosbodytrack` inventory subtrees, then
selects `videosoriginal/<subject>/<recording>.mp4` explicitly. The selected path
is an original-subtree camera candidate until a decoder validates the media;
the optional per-record `bodytrack_qa_video_path` cannot satisfy source
completeness or substitute for `camera_video_path`. The hash-bound
[v2 release audit](../../benchmarks/vidimu_v2_release_audit.json), generated by
the checked-in [audit command](../../benchmarks/audit_vidimu_v2_release.py),
reports same-stem original candidates for all 208 paired CSV/RAW records in
both video archives, whereas only 206 have a same-stem rendered BodyTrack
video.

## VIDIMU PTS/CV finalization v0.3

The frame-finalization layer is a separate pre-synchronization artifact. It is
not added to the seven-table TremoraStore v0.1 snapshot, because doing so would
require invented IMU clocks, alignment rows, and windows. One immutable
finalization identity is published at:

```text
finalized/<recording_id>/<finalization_id>/
├── video_frames.parquet
├── cv_frame_results.parquet
├── cv_detections.parquet
├── finalization_manifest.json
├── finalization_audit.json
└── _SUCCESS
```

For Gate B, that exact finalization-ID namespace has one second terminal form:

```text
finalized/<recording_id>/<finalization_id>/
├── source_failure_manifest.json
├── source_failure_audit.json
└── _FAILURE
```

The two forms are mutually exclusive: atomic no-replace publication prevents a
success and a source-failure outcome from occupying the same frozen processing
identity. `_FAILURE` is not a frame/CV bundle and makes no successful decode,
inference, synchronization, or window claim.

`PTSDecoder` uses PyAV 18 and its linked FFmpeg libraries as the timestamp
authority. It hashes and decodes the same no-follow descriptor, records raw PTS
and time base, applies a pure 0/90/180/270-degree display matrix exactly once,
and never derives time from nominal FPS or OpenCV position fields. Its
`decode_ordinal` means decoder-emission order, not coded-packet order. Valid PTS
frame identities use the source video hash, absolute stream index, raw PTS, and
duplicate rank. A missing-PTS frame is retained in a separate identity namespace
using the frozen decoder-emission ordinal and remains temporally ineligible.

The versioned association contract is
`tremora-pose-frame-association-1.0.0`. The decoder creates each `frame_id`; the
offline estimator consumes it. Every decoded frame has exactly one
`cv_frame_results` row, while `cv_detections` contains zero or more rows. A
no-detection or inference-failure frame therefore cannot disappear. Landmark,
bbox, transform, and coordinate-space fields use fixed Arrow structures rather
than JSON. Optional fixed-size model arrays represent absence with all-null
elements; they are never zero-filled. Canonical `runtime_ms` is null so wall
time cannot break deterministic artifact bytes.

Gate A uses generated CFR, VFR, B-picture, non-zero-start, missing/duplicate/
non-monotonic PTS, discontinuity, rotation, blank/multi-target, and damaged-file
fixtures with an injected deterministic estimator. A release `PASS` requires a
second root with distinct artifact inodes whose complete bundle bytes match.
The audit proves byte-identical stored artifacts across distinct trees; it does
not by itself attest how the second tree was produced. Gate A's tests construct
both trees through separate finalization calls. Gate B is fail-closed: the
snapshot preflight requires the exact frozen recording inventory, complete
VIDIMU `dataset.zip` archive, exactly one selected video archive, original
videos, paired IMU files, license record, and every corresponding hash before
any record is processed. Every Gate-B manifest carries the same canonical
`source_archives` evidence: one `DATASET_ARCHIVE` and one `VIDEO_ARCHIVE`, each
bound by safe original path, role, and SHA-256; local materialization paths are
never persisted. Its release audit additionally requires caller-supplied frozen
anchors for dataset ID/version, inventory and license hashes, and both archives'
path/role/hash tuples. Decoder and estimator factories must return a fresh,
reset instance for every recording so state cannot make snapshot order or
resume behavior affect output. The estimator remains an injected,
hash/version-bearing interface; the live single-hand, stateful MediaPipe worker
is not silently treated as the required offline multi-detection configuration.

After complete Gate-B preflight has verified every frozen asset, a video whose
pinned descriptor and expected SHA-256 both verify may still be rejected by the
trusted PyAV/FFmpeg media decoder. Only the decoder's closed, post-verification
media-rejection signal can publish the three-file `_FAILURE` outcome. Missing,
nonregular, substituted, hash-mismatched, concurrently changed, wrongly
configured, out-of-memory, environment, factory, provenance,
processing-identity, or other non-frame system faults remain snapshot `NO-GO`;
exception text and local paths are never stored as failure identity. Frame-local
preprocessing and inference failures instead remain explicit
`PREPROCESS_FAILURE` or `INFERENCE_FAILURE` result rows in a successful bundle.
The factory's exact `DecodeConfig` is reconstructed into a fresh trusted base
`PTSDecoder`, so subclass and per-instance method overrides cannot forge
eligible failure evidence.

A Gate-B release `PASS` requires every frozen recording to have exactly one
strictly audited outcome in each of two distinct byte-identical artifact trees.
`videos_opened` equals the frozen recording inventory; `videos_failed` is the
documented source-decode-failure subset. Decoded-frame/CV reconciliation and
all frame-level counts are computed only from successful bundles, while the
release audit independently rehashes the video, paired IMU, both archives,
inventory, and license for every outcome.

The strict per-record and release audit entry points are:

```bash
cd detector
python -m motionbloom.tremora_store.finalize.audit_finalized_recording \
  /absolute/path/to/finalized/<recording_id>/<finalization_id>

python -m benchmarks.audit_vidimu_pts_cv_release \
  --finalized-root /absolute/path/to/first-root \
  --replay-root /absolute/path/to/distinct-artifact-replay-root \
  --recording-ids /absolute/path/to/frozen-recording-ids.json \
  --required-gate GATE_A_SYNTHETIC
```

Gate B also requires `--source-root`, `--inventory-manifest`,
`--license-record`, dataset ID/version, inventory/license SHA-256 anchors, and
all four role-bound archive arguments:
`--dataset-archive-path`, `--dataset-archive-sha256`,
`--video-archive-path`, and `--video-archive-sha256`.

No Gate-B VIDIMU finalization artifact is checked in. A Gate-A contract `PASS`
does not imply that the public VIDIMU snapshot, the production estimator, or
video–IMU synchronization has passed.

Run this from the repository's Python environment after installing the detector
storage dependencies (including PyArrow). From the repository root, reproduce
the checked artifact with the pinned `dataset.zip` and live Zenodo
metadata/range evidence:

```bash
cd detector
python -m benchmarks.audit_vidimu_v2_release \
  --dataset-zip /absolute/path/to/dataset.zip \
  --output benchmarks/vidimu_v2_release_audit.json
```

Secure audit publication requires POSIX `dir_fd`, `O_DIRECTORY`, and
`O_NOFOLLOW` support and deliberately fails closed where those primitives are
unavailable, including the current Python Windows implementation. The checked
JSON is canonical UTF-8 with LF line endings; preserve LF for
`benchmarks/vidimu_v2_release_audit.json` when comparing artifact bytes (for
example, do not subject that file to automatic CRLF checkout conversion).

The source adapter parses the exact released pose CSV and RAW formats but has no
cross-modal snapshot publication method. Pose CSV rows contain 34 XYZ BodyTrack
positions in millimetres, including the release's three leading-space
right-thumb header tokens. They contain no timestamp, frame ID, confidence, or
visibility field.
The parser therefore emits an intermediate `pose34` source table keyed only by
row ordinal; exact zero triplets are reported as an observed sentinel pattern,
not promoted to a tracking-validity decision. The PTS/CV finalizer does not bind
these BodyTrack rows by count or position; they remain unassociated evidence.

RAW rows contain scalar-first WXYZ quaternion observations only—never released
accelerometer, gyroscope, or magnetometer axes. The first five rows are
positional N-pose calibration observations in the documented activity-specific
sensor order. They are recognized by position, not timestamp: 197 of the 208
released records use timestamp zero, while 11 S41 records use nonzero
large decimal values. Every source row and held value is retained in a dedicated
unit-unknown RAW source table. Invalid non-unit measurements are preserved in
the source audit record and represented as null quaternion channels with
`INVALID_IMU_PAYLOAD`; the 12 audited invalid observations in `S54_A08_T02` are
neither normalized nor dropped.

The paper documents nominal 50 Hz sensor output, but the released RAW timestamp
field is not documented as a clock source, unit, or video relation. The audited
files contain a much faster row cadence with many held quaternion values. The
parser preserves the exact decimal timestamp token and uses context-independent
integer string parsing only to detect equality, reversal, and whether dynamic
rows follow calibration. It emits no `sensor_time_native_ns`,
`canonical_time_ns`, `imu_samples`, or `clock_map` values. The capture is marked
`UNRESOLVED_TIMESTAMP_UNIT_CLOCK_SOURCE_AND_VIDEO_RELATION`, and canonical
materialization is `DEFERRED`. Row cadence is not a sensor-bandwidth claim or a
50 Hz reconstruction. Neither the audited release files nor the documented
synchronization workflow establish a RAW-token unit, clock source, or relation
to video PTS. No workflow frame-cut or RMSE value is therefore promoted to a
clock residual in milliseconds or a `VALID` affine map.

Matching `.mp4`/`.csv`/`.raw` stems still do not establish alignment. The Gate-A
PTS decoder and CV frame-association contract are implemented, but a complete
real VIDIMU finalization, synchronization, accepted continuity, and
frequency-eligible VIDIMU windows remain unvalidated and blocked. Ego4D and
controlled corruptions remain stress inputs, not
independent clock ground truth. SHA-256 digests for both downloaded archives and
the selected extraction scope remain mandatory; caller-supplied archive digests
do not prove complete extraction.

Adapter-mediated RAW/pose parsing pins extraction and subject-directory
identities, opens each path component relative to trusted no-follow directory
descriptors, and binds parsed bytes to unchanged before/after file metadata. It
fails closed on platforms without that POSIX `openat` capability; the v0.1
implementation does not claim equivalent ancestor-swap resistance from
Python's Windows filesystem API.

Official N-pose, BodyTrack-rendered, `.sto`/`.mot`/`ik_*.mot`, IK error, and
fullsize `.mp4.out` companions are never paired as original source inputs. The 13
mixed-trial S48 `.mot` paths and the v2 S41 `P01`, S49 `T01V2`, and misplaced
S25-under-S24 residues remain exact path-bound exclusions; they do not broaden
the canonical `Txx` grammar or replace a canonical source. Selected subtree
components and entries must not be symlinks, and unexpected case, delimiter,
nesting, subject, or filename variants fail closed.
