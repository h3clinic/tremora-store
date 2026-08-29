# TremoraStore Dataset Architecture v0.2

**Status:** authoritative dataset-role and timing-authority scope. Supplements
`TremoraStore_Public_Data_Benchmark_Spec_v0_1.md`, which remains in force for
storage contracts, table semantics, the anomaly matrix, baselines and go/no-go
gates.

**Provisional title:** *TremoraStore: Authority-Aware Storage and Deterministic
Replay for Video–IMU Motor Analytics*

## 1. Why this version exists

VIDIMU is finished as a synchronization substrate. This is a closure, not a
pause, and it is not reopened by a new parser, a clearer grouping rule, or
another salvage contract. The published structures are mutually insufficient:

- RAW polling groups cannot be mapped authoritatively to STO/MOT ordinals;
- most tested RAW trims cut through five-sensor group boundaries;
- the published source transformation never modifies the MP4.

Any decoded-frame-to-IMU relationship would therefore be invented by
TremoraStore rather than preserved from the source.

The 217/217 byte-identical replays recorded in the v0.5D audit prove that the
source procedure is reproducible. They do not prove that the procedure
establishes a defensible video–IMU timeline. That distinction is the finding,
and it belongs in the paper as a first-class result rather than as an apology.

v0.5 (`NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`) and v0.5D (source-derived
materialization `NO_GO`) are permanent. Nothing here revises, supersedes or
reinterprets either report, and the checked JSON for both is hash-pinned by
`detector/tests/test_timing_authority_contract.py`.

## 2. Central contribution

A multimodal storage architecture that explicitly represents timing authority,
preserves source-native evidence, refuses unsupported synchronization, and
materializes deterministic frame–IMU indexes only when the source provides an
admissible temporal contract.

## 3. Timing-authority model

Implemented in `motionbloom/tremora_store/timing_authority.py`. The tiers are
frozen, ordered strongest to weakest, and mutually exclusive:

```text
RAW_SHARED_CLOCK
RAW_MAPPED_CLOCK
SOURCE_CANONICAL_TIMESTAMP
SOURCE_ALIGNED_RELATIVE_TIME
SOURCE_DERIVED_ALIGNMENT
SOURCE_RELATIVE_UNIMODAL_CLOCK
AMBIGUOUS
UNRESOLVED
```

Rules the code enforces rather than documents:

- A frame-to-IMU index requires one of the first five tiers.
- A storage benchmark additionally admits `SOURCE_RELATIVE_UNIMODAL_CLOCK`.
- `AMBIGUOUS` and `UNRESOLVED` admit neither.
- Only `RAW_SHARED_CLOCK` may support a hardware-synchronization claim.
- A binding that declares a derived tier must state the assumption it derives
  under; an unstated assumption is a construction error. A tier that resolves
  nothing cannot state one.

VIDIMU binds at `UNRESOLVED`, so excluding it from paired indexing is a type
error rather than a policy. The VIDIMU-specific `AlignmentAuthority` enum in
`v05d/authority.py` is frozen evidence and is not edited to accommodate another
dataset; the cross-dataset model generalizes the decision it forced.

## 4. Three-dataset architecture

| Dataset | Role | Timing authority |
|---|---|---|
| VIDIMU | Fail-closed case study | `UNRESOLVED` |
| Ego4D IMU | Paired timing, indexing, storage and replay benchmark | `SOURCE_CANONICAL_TIMESTAMP` |
| PADS | Tremor-specific high-rate IMU workload | `SOURCE_RELATIVE_UNIMODAL_CLOCK` |

### 4.1 Ego4D — next primary substrate

The Ego4D IMU documentation describes normalized IMU CSVs with the source-order
columns `component_idx`, `component_timestamp_ms`, `canonical_timestamp_ms`,
`gyro_x/y/z`, `accl_x/y/z`. Acceleration is spelled `accl_*` in the source and
TremoraStore preserves that spelling: renaming a source column is a silent
claim that the two are interchangeable.

Canonical timestamps are offsets with respect to where each video component
starts in the canonical video, which makes them directly usable for frame-range
indexing. The documentation also discloses the limits: the first IMU timestamp
is assumed aligned to the original container's `t = 0`; some canonical
timestamps are null because canonical videos are trimmed to the video-stream
region; some videos lack IMU for every component; some files lack complete
acceleration; some timestamps are significantly large, small or non-monotonic;
and no IMU calibration was performed.

That combination makes Ego4D suitable for an authority-aware, gap-aware storage
system and unsuitable for a raw hardware-clock synchronization claim. Ego4D is
bound to `SOURCE_CANONICAL_TIMESTAMP` and may not be relabelled
`RAW_MAPPED_CLOCK`; `ego4d.authority.assert_not_relabelled` refuses it.

Ego4D's own guidance is to sort by canonical timestamp before use. That is a
reading convenience and belongs in a query view. The persisted authority table
keeps source order, so a non-monotonic row stays visible where the source put
it.

### 4.2 PADS — the tremor workload

The PADS PhysioNet release contains 5,159 assessment steps from 469
participants across 11 neurologist-designed movement tasks, with bilateral
wrist acceleration and rotation, under CC BY 4.0. There is no paired video.

PADS therefore cannot validate cross-modal alignment, and the code makes that
structural rather than advisory. Every PADS table and field name is screened
for video-bearing substrings — `video`, `camera`, `frame`, `rgb`, `pts`,
`pixel` — rather than checked against a list of exact names: a deny-list would
refuse `video_uid` while admitting `video_uid_ref` or `camera_stream_uid`.
`assert_no_paired_claim` refuses a cross-modal assertion outright.

PADS publishes a source `Time` channel. Each observation record declares seven
columns per device — `Time` in seconds followed by `Accelerometer_X/Y/Z` in g
and `Gyroscope_X/Y/Z` in rad/s — alongside a separately declared
`sampling_rate` of 100. The source time column is therefore the timeline, and
every sample records `relative_time_basis = SOURCE_TIME_COLUMN`. The declared
rate is a validation constraint on that timeline, not a generator for it: an
invalid time value is never replaced by `sample_ordinal / 100`. The record's
gate closes instead.

Recording length comes from the release, never from task names:

```python
expected_row_count = session["rows"]
```

`observation_001.json` declares 2048 rows for `Relaxed`, `RelaxedTask` and
`Entrainment`, and 1024 for the eight tasks between them. All are whole
recordings. A name-based rule gets `Entrainment` wrong — it is the last session
and easily mistaken for one of the short ones — and a hardcoded 2048 would
reject eight of the eleven published tasks as split recordings. Only a file
whose length disagrees with its own declaration is a fragment. A test asserts
that neither `2048` nor `1024` appears in the parser or the audit engine.

Two timing quantities are kept apart, because they differ by exactly one sample
period:

```text
sample support duration  = rows / rate
first-to-last time span  = (rows - 1) / rate
```

So 2048 samples at 100 Hz support 20.48 s but span 20.47 s, and 1024 samples
support 10.24 s but span 10.23 s. Requiring `t_last - t_first == 20.48` would
be an off-by-one-sample error, and a recording that does span 20.48 s has one
sample too many.

**The published clock is a real device clock, not a nominal grid.** In
`001_Relaxed_LeftWrist.txt` the intervals run from 7.13 ms to 12.90 ms around a
9.99 ms median, and the file spans 20.459 s where exact 100 Hz would give
20.47 s. A strict per-interval or exact-span check would therefore fail every
file in the release. Only the median interval is compared against the declared
period, and both the cadence and span comparisons use frozen relative
tolerances declared in `pads/movement.py`.

Channel order comes from each observation record, and the declaration is
followed rather than overruled. The release stores one file per device per
session, seven columns, not twelve. Physical column *i* is read as declared
channel *i*, and the result is persisted in TremoraStore's internal canonical
order alongside `source_channel_order`, `source_units_order` and the
`canonicalization_permutation` that relates them, so the reordering is explicit
and reversible. A permuted declaration is therefore an issue to report, not a
refusal: the limb is identified at record level by `device_location`, so a
permutation cannot relabel a limb, and it could only mislabel a sensor if the
parser ignored the declaration and assumed fixed physical columns — which is
exactly what it does not do.

The gate closes only on genuine ambiguity, where no reading of the file follows
from what the metadata says: a duplicate, missing or unknown channel name, a
unit that contradicts its named channel, channel and unit lists of different
lengths, a missing or unrecognized `device_location`, or a row whose column
count disagrees with the declaration. Units are checked per channel *name*, so
a reordered declaration is still verified.

A blank source row is a parse failure rather than a skipped line; a record with
no usable sensor value is refused as `NO_USABLE_VALUES`, so a corpus with no
signal cannot satisfy the gate on row counts alone; and neither the observation
index nor the metadata may name a file outside the movement root.

Derived representations at 100/50/30/25 Hz must each declare an anti-aliasing
treatment; a bare decimation is refused. At 25 Hz the rate-based ceiling is
12.5 Hz against a 12 Hz upper tremor bound — it clears the arithmetic by
0.5 Hz, and the declaration records that rather than presenting it as headroom.
None of that is opened by P0.1.

### 4.3 EgoInertia-MI — verified, optional, off the critical path

**Correction.** An earlier pass recorded this dataset as `UNVERIFIED_SOURCE`
and excluded it as probably fabricated. That was wrong: it was a search
failure, not a fact. arXiv:2607.03934 resolves to *EgoInertia-MI: A Multimodal
Egocentric Vision and IMU Benchmark for Motor Impairment Assessment*
(Alhamdoosh, Pala, Mohamed, Arvind), submitted 4 July 2026, with a project page
at `fatemah-alh.github.io/EgoInertia-MI-Page/`.

The abstract describes synchronized egocentric video and wearable IMU across 19
upper- and lower-body activities, performed by healthy volunteers simulating
three severity levels (none, mild, severe), with action-recognition and
severity-estimation baselines.

Its participants simulate impairment, so it can support motor-analytics systems
testing but not clinical validation, and it enters through the same audit path
as any other dataset — a stated authority tier and a gate — if and when it is
used. It must never block the core paper.

## 5. E4D-P0.1 — Ego4D source-canonical timing-authority audit

### 5.1 Objective

Determine exactly which Ego4D IMU rows and video components can support a
source-authoritative canonical timeline. No frame-to-IMU index, window,
spectrum or performance benchmark is created in P0.1, and gate condition 11
fails the audit if one is emitted.

### 5.2 Authority contract

```text
timing_authority        = SOURCE_CANONICAL_TIMESTAMP
raw_shared_clock        = false
hardware_sync_claim     = false
derived_under_assumption = true
```

### 5.3 Tables

Four Parquet tables, defined in `ego4d/schemas.py`:

- `ego4d_imu_assets` — one row per considered (video, component) asset triple.
- `ego4d_imu_authority_rows` — one row per source IMU row, in source order,
  carrying both the original timestamp tokens as strings and their parsed
  numeric values.
- `ego4d_video_timeline_authority` — component placement on the canonical
  timeline plus the PTS-derived timeline it is checked against.
- `ego4d_timing_authority_summary` — one additive accounting row per video.

### 5.4 Row statuses and issue bits

`canonical_authority_status` is a single-valued verdict resolved by a frozen
precedence chain, highest priority first:

```text
COMPONENT_NOT_COVERED
SOURCE_CANONICAL_NONFINITE
SOURCE_CANONICAL_NULL_AFTER_TRIM
SOURCE_CANONICAL_OUTSIDE_VIDEO
SOURCE_CANONICAL_NONMONOTONIC
SOURCE_CANONICAL_DUPLICATE
SOURCE_CANONICAL_EXTREME_MAGNITUDE
SOURCE_CANONICAL_UNPARSEABLE_TOKEN
MISSING_ACCELERATION
MISSING_GYROSCOPE
SOURCE_CANONICAL_VALID
```

Because one row can exhibit several conditions and a single verdict can show
only one, every observed condition is additionally retained in `issue_bits`.
Precedence decides what a row is *called*; it never decides what is *recorded*.

VALID and eligible are the same set by construction. Every disqualifying issue
bit has an entry in the precedence chain, and an import-time assertion fails the
module if one does not, so a row the audit will not build an index on can never
be labelled valid or counted in `canonical_rows_valid`.

A token is parsed only if it is a plain decimal, a recognized null, or a
recognized non-finite spelling. Python's `float` would accept `1_000` and store
a number the source never wrote while the preserved token still said `1_000`;
those are recorded `SOURCE_CANONICAL_UNPARSEABLE_TOKEN` instead. Null and
non-finite are different facts and are matched case-insensitively, so `NaN` and
`NAN` cannot land in different buckets.

Non-monotonicity is measured against the monotone frontier already established
in source order, not merely the previous row. Comparing against the previous row
would let a value below the frontier re-enter the eligible set as soon as one
lower value preceded it, and the eligible subsequence would stop increasing.

A blank record anywhere but a single trailing terminator is a parse failure.
Skipping one would delete a source row and shift every ordinal after it.

The parser does not infer a null canonical timestamp, replace a timestamp using
the sample rate, repair an extreme timestamp, discard a non-monotonic row, or
overwrite source order with canonical-time order.

### 5.5 Benchmark subset

The subset is frozen before any storage performance is measured, from metadata
alone, as a pure function of

```text
selection_seed              = 20260828
selection_algorithm_version = tremora-ego4d-stratified-selection-0.1.0
metadata_snapshot_sha256
```

No random number generator is involved: candidate order comes from a keyed
SHA-256 digest, so two clean roots agree without shared state and input order
cannot change the result. Strata:

```text
CLEAN_MONOTONIC
NONMONOTONIC_SOURCE_ORDER
NULL_CANONICAL_TIMES
PARTIAL_COMPONENT_COVERAGE
MISSING_ACCELERATION
EXTREME_TIMESTAMP
```

Floors: at least 100 videos, at least 10 hours of canonical video–IMU overlap,
at least two capture-device groups, and representation of every available issue
stratum. A stratum the source cannot supply is named in
`strata_absent_in_source` and given a non-zero shortfall; it is never topped up
from another stratum, because a fabricated stratum destroys the meaning of the
stratification. The floors actually applied are published inside the evidence
block, so a run that lowered them cannot look like one that did not.

Paired overlap is the canonical time actually covered by authority-eligible IMU
rows, clamped to the video, measured relative to each component's own cadence.
`dt_ref` is the median positive finite interval between consecutive eligible
samples, estimated only when there are at least eight such deltas — fewer is not
a cadence, and the component then reports no coverage at all. Each eligible
sample supports `[t - dt_ref/2, t + dt_ref/2]`, and coverage is the length of
the union of those supports, clamped to the video interval.

Three weaker definitions were tried and all fail the same way. The video's
duration lets a ten-minute video whose IMU covers two hundred milliseconds
contribute ten minutes. The first-to-last span lets a two-row file contribute a
full hour. A flat 100 ms bridge is better but still too generous: Ego4D's own
documented example is spaced ~4.975 ms, about 201 Hz, so 100 ms spans roughly
twenty expected sample intervals. Under sample support, two samples an hour
apart contribute about two sample intervals.

Continuity is a separate question from coverage. A segment breaks when
`Δt > min(100 ms, 3 × dt_ref)`, which at the documented cadence is ~14.9 ms. The
multiplier is project policy, not a fact Ego4D supplies: it is frozen at 3 and
its behaviour at 2×, 3× and 5× is exercised by the test suite. The 100 ms value
survives only as an absolute ceiling, so a pathologically low-rate component
cannot bridge an arbitrarily large interval either.

### 5.6 Validation

For each selected video: decode PTS with the already validated PTS-preserving
decoder, preserve the exact frame timeline, read the source
`canonical_timestamp_ms`, verify each valid IMU timestamp against the canonical
presentation interval, quantify nearest-frame and containing-interval
relationships without persisting them, and confirm that PTS-derived video time
and Ego4D canonical time share the declared origin.

`ego4d/pts_validation.py` returns those quantities in memory only. Persisting
them would be a P0.2 index, which P0.1 forbids.

### 5.7 Hard gate

Eleven conditions, all required:

```text
ALL_ASSETS_HASH_VERIFIED
EVERY_IMU_ROW_REPRESENTED
SOURCE_TIMESTAMP_TOKENS_PRESERVED
NO_MISSING_TIMESTAMP_INFERRED
EVERY_SELECTED_VIDEO_HAS_PTS_TIMELINE
VALID_CANONICAL_ROWS_INSIDE_VIDEO_INTERVAL
KNOWN_ISSUE_ROWS_VISIBLE_AND_CLASSIFIED
NO_ROW_DROPPED_FOR_NON_MONOTONICITY
AUDIT_REPRODUCES_BYTE_IDENTICALLY
SUBSET_FLOORS_SATISFIED
NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED
```

Gate status is `PASS_SOURCE_CANONICAL_TIMESTAMP_AUTHORITY` or
`NO_GO_EGO4D_CANONICAL_TIMESTAMP_AUTHORITY`. The audit is allowed to return
another no-go. If the official canonical timestamps cannot be reconciled with
video PTS, no rescue mapping is constructed.

Three conditions are only as strong as what they are measured against, and each
is checked against something the audit did not derive from the thing under test:

1. Row representation compares the authority rows against a data-line count
   taken from the split records before any row object exists. A check that
   compares a list against its own length can only ever pass.
2. Row loss is checked as a gapless zero-anchored ordinal sequence over the
   whole file. Anchoring on the first surviving ordinal, or grouping by
   component, hides a lost prefix, a lost suffix or a lost whole component —
   and, because a normalized Ego4D CSV may interleave components, grouping by
   component also raises a false no-go on a perfectly intact file. Loss at the
   end leaves a gapless sequence, which is why the row-count condition exists
   alongside this one.
3. Reproducibility requires a second report that identifies itself as a P0.1
   record of the same schema, produced by the same implementation hashes, over
   the same metadata snapshot, **and** naming a different publication
   destination. A bare evidence hash is not evidence that another root ran, and
   identity fields alone do not distinguish a second execution from this run's
   own record copied to a new filename.

Origin agreement is likewise two tests, not one, and the component's timeline
status names which failed. The origin test asks whether the first decoded frame
sits at canonical zero; the span test asks whether the decoded timeline is as
long as the source says the canonical video is. A span comparison alone cannot
see a shifted origin, because a constant shift cancels out of a span. The span
tolerance is one frame interval plus a slack, with the interval term capped:
uncapped, a two-frame timeline would grant itself a tolerance proportional to
its own length and a one-hour decode would "agree" with a half-hour video.

Like the v0.5D execution receipts, the reproduction check establishes two
executions under the trusted procedure; it is not cryptographic remote
attestation and does not defend against a deliberately forged record.

Audit-execution status and gate status are separate, exactly as in v0.5D: a
successful audit that closes the gate reports `PASS` execution and a `NO_GO`
gate, and exits 3.

### 5.8 Release behaviour with no data

An audit whose dataset is not present emits
`release_status = BLOCKED_INPUT_DATA_UNAVAILABLE`, `gate_evaluated = false`, no
gate status, no evidence hash, no result table, no success marker, no
frame-to-IMU index and no window, and exits 4 — distinct from 3, so a caller can
tell "we audited and it failed" from "we have not got the data". The claim
boundary is asserted even on that record: it must not be the one artifact that
escapes the authority checks.

Blocked means there was nothing to audit. It is deliberately not a test of which
flags were passed, and not a description of malformed data. An empty asset
manifest, an absent video root, a file that fails to parse, and an index entry
that tries to escape its root are all evidence about a release, and evidence
closes a gate. Reporting any of them as unavailability would hide a bad release
behind an availability notice — the same inversion, in the opposite direction,
as publishing a gate outcome with no data behind it.

## 6. PADS-P0.1 — source-relative unimodal-clock ingest audit

Fourteen conditions, all required:

```text
PADS_ALL_SOURCE_FILES_HASH_VERIFIED
PADS_RELEASE_STRUCTURE_RECONCILED
PADS_EVERY_DECLARED_STREAM_PARSED
PADS_ROW_COUNTS_METADATA_DIRECTED
PADS_SOURCE_TIME_IS_THE_TIMELINE
PADS_CADENCE_AGREES_WITH_DECLARED_RATE
PADS_SOURCE_ORDER_AND_UNITS_PRESERVED
PADS_NO_AMBIGUOUS_DECLARATION
PADS_DEVICE_LOCATIONS_RECOGNIZED
PADS_NO_BLANK_ROW_DISCARDED
PADS_USABLE_VALUES_PRESENT
PADS_NO_VIDEO_ASSOCIATION_EMITTED
PADS_INDEPENDENT_REPRODUCTION_VERIFIED
PADS_NO_INDEX_OR_WINDOW_ARTIFACT_EMITTED
```

Gate status is `PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK` or
`NO_GO_PADS_UNIMODAL_INGEST`.

Two of these bind the release rather than a single file, and both were part of
the contract before the first authoritative run.

**Release structure.** The reconciler proves 469 participants × 11 tasks = 5,159
assessment steps × 2 wrists = 10,318 device files from the source metadata
rather than assuming it. It fails on a missing or duplicated task, a missing or
duplicated wrist record, an unknown extra task, a missing or traversing file
reference, a reference that does not name its own participant/task/wrist, or any
release-level count mismatch. The expected task names and totals are frozen;
individual sample counts stay metadata-directed.

**Independent reproduction.** Output is two layers. The evidence record contains
only source-derived canonical content, so two genuine executions produce
byte-identical evidence. Everything execution-specific — run id, process id,
output root, output-root inode, command arguments — lives in a separate receipt.
The verifier requires the two receipts to agree about the release, the
implementation and the evidence hash while disagreeing about where and by whom
they ran. A report copied to a second path satisfies every identity field, so
identity alone cannot distinguish a second execution from `cp`. The release
driver spawns both audits itself, in two child processes against two empty
output roots, rather than accepting two operator-supplied report paths. As with
v0.5D, this is not remote attestation.

`RECOGNIZED_DEVICE_LOCATIONS` is `{LeftWrist, RightWrist}`, verified against all
469 published observation records. A third location would close the affected
records' gate as `UNRECOGNIZED_DEVICE_LOCATION` rather than parse; widening the
set is a separately versioned contract change.

## 7. Milestones after P0.1

**E4D-P0.2 — frame-to-IMU range index.** Opens only on a passing P0.1 gate. For
each frame interval `[t_i, t_(i+1))`, store the contiguous range of eligible IMU
rows whose source canonical timestamps fall inside it. The index references IMU
rows; it never copies their sensor values.

**E4D-P0.3 — gap-aware windows.** Four-second windows that cannot cross
component boundaries, null canonical spans, timestamp discontinuities,
uncovered components, or invalid frame intervals.

**E4D-P0.4 — storage and retrieval benchmarks.** B0 CSV + runtime timestamp
join; B1 per-frame JSON with duplicated IMU samples; B2 monolithic HDF5; M1
columnar streams + frame-range index. Measured on bytes per hour, duplicated IMU
values, index overhead, random four-second retrieval p50/p95, sequential replay
throughput, batch loading, peak memory and deterministic replay hashes. The
v0.1 baseline fairness rules apply unchanged.

**PADS-P0.2 — tremor workload.** Opens only on a passing PADS-P0.1 gate:
immutable participant/task/stream indexes, exact task-local sample-range
retrieval, four-second gap-aware windows, participant-disjoint grouping and
deterministic replay. Spectral features and resampling ablations are a separate
milestone and are not combined into the ingest work.

## 8. What the paper says about VIDIMU

> Of the evaluated multimodal source structures, reproducibility of a
> publisher-provided transformation was not sufficient to establish temporal
> authority. TremoraStore reproduced all source transformations byte-for-byte
> but withheld aligned materialization because raw polling groups, derived
> motion ordinals, and decoded frames lacked a defensible common mapping.

That is not an embarrassing failure. It is the direct evidence for why the
authority model is necessary.

## 9. Provenance of this branch

The E4D-P0.1 and PADS-P0.1 implementation on this branch is a **re-derivation**.
An earlier build of the same design was produced in an ephemeral environment and
lost with it, together with the adversarial reviews it had passed. This code was
rebuilt from the architecture and has not carried those reviews. It is a new
artifact and the commit messages say so.

## 10. Current status

| Component | Status |
|---|---|
| Cross-dataset timing-authority model | Implemented and frozen; adversarial unit tests only |
| VIDIMU v0.4 Gate B | `PASS`; unchanged |
| VIDIMU v0.5 / v0.5D | `NO-GO`; permanent, unchanged, hash-pinned |
| E4D-P0.1 contract, schemas, parsers, subset, gate | Implemented; not executed against Ego4D assets — the CLI returns `BLOCKED_INPUT_DATA_UNAVAILABLE` |
| E4D-P0.1 empirical gate | Not evaluated; requires the signed Ego4D licence and pinned assets |
| E4D-P0.2 / P0.3 / P0.4 | Closed pending a passing P0.1 gate |
| PADS-P0.1 contract, schemas, parser, audit | Implemented and executed against PADS 1.0.0 |
| PADS-P0.1 empirical gate | See `pads_p01_release_audit.json` |
| PADS storage/spectral benchmarks | Closed pending a passing PADS-P0.1 gate |
| EgoInertia-MI | Verified real (arXiv:2607.03934); optional, off the critical path |
| Comparative systems results | None yet |
