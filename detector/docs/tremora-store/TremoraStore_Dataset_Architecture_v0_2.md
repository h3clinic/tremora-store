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

## 7. PADS-P0.2 — authoritative indexing and gap-aware replay

### 7.1 What it materializes, and what it refuses to

P0.2 materializes a compact deterministic representation supporting participant
lookup, assessment and task lookup, left- and right-wrist stream retrieval,
exact task-local sample ranges, gap-aware contiguous segments, four-second
window indexes, participant-disjoint fold assignment and byte-exact replay.

It calculates no spectrum, tremor frequency, band power, resampled signal,
anti-aliasing output, classification, video association or comparative
benchmark result.  Those are P0.3, P0.4 and P0.5.  Every table and field name
passes two substring screens — the inherited video screen and a P0.2 screen for
spectral, resampling and classification names — and the audit checks the screen
against the files it actually wrote, not against its own prose.

The success marker is `_PADS_P02_INDEX_SUCCESS`, never a generic `_SUCCESS`: an
index materialization must not be mistakable for a synchronization result.

### 7.2 The P0.1 dependency

P0.2 refuses to run without the exact P0.1 authority, pinned in code and
generated into `pads_p01_dependency.json`, so editing the file alone cannot
move the dependency.  P0.2 never regenerates P0.1 inside itself.

Absence and disagreement are separated the way the project separates them
elsewhere.  A missing dependency, report or release root means there is nothing
to depend on: `BLOCKED_P01_DEPENDENCY_UNAVAILABLE`, exit 4.  A changed evidence
hash, a changed report, a P0.1 verdict that is not `PASS`, or a different
source manifest is evidence about a broken authority chain: the gate is
evaluated, `P01_AUTHORITY_DEPENDENCY_VERIFIED` fails, nothing is materialized,
and the run exits 3.  Reporting a disagreement as unavailability would hide a
broken authority chain behind an availability notice.

### 7.3 Time is exact only in picoseconds

Every `Time` token in the release carries exactly ten decimal places, so its
resolution is 1e-10 s.  An integer nanosecond count cannot represent that:
`0.0099029541` s is 9,902,954.1 ns but exactly 9,902,954,100 ps.  Every
source-derived time in P0.2 is therefore an int64 picosecond count, a schema
test rejects any field named `_ns`, and a token finer than the declared scale
raises rather than rounding.

Task-local time is a convenience view derived from an explicitly stored origin;
source time remains the authority.

Sensor values are stored as float64 and their exact source token is rebuilt
through a declared `{:.10f}` format — but the round-trip is verified for every
value as it is materialized and the stream is refused on the first value that
does not rebuild its token.  That is what makes byte-exact replay a proven
property of this corpus rather than an assumption about it.

### 7.4 The bilateral boundary

The release establishes that two wrist streams belong to the same participant
and task.  It establishes nothing about a common hardware clock, so:

```text
bilateral_pairing_authority           = SOURCE_PROTOCOL_PAIR
cross_wrist_clock_alignment           = UNRESOLVED
sample_level_bilateral_fusion_allowed = false
```

P0.2 may retrieve both wrists for one task, and windows are paired by their
shared task-local grid offset.  It may never claim that left sample 400 is
simultaneous with right sample 400.  Every published bilateral row carries
`UNRESOLVED` on its own row rather than leaving it to a sibling table.

### 7.5 Storage

Every sample is stored exactly once, packed by stream: streams in ascending
`stream_id` order, 256 per part file, exactly one row group per stream so a
stream is one contiguous read, under one pinned writer configuration.  10,318
tiny text files are not a runtime representation, and copying samples into task
or window records would store the corpus several times over.

### 7.6 Segments and windows

Segments are recomputed from the stored samples every time rather than trusted
from P0.1, and break at `min(100 ms, 3 x dt_ref)` — about 30 ms at the
release's ~9.99 ms median interval.  Breaks also fire on a non-positive delta,
an ordinal discontinuity and an invalid sample, and the result is checked to
partition its stream exactly.

Windows sit on a 2 s grid anchored at task-local zero and span 4 s.  Membership
is a half-open source-time interval searched inside the owning segment, never a
fixed forward count: the device clock jitters from 13.8 us to 58.8 ms across the
corpus, so `first_sample + 400` would silently mean a different duration in
every window.  Containment is verified on the built windows, not assumed from
the construction that made them.

### 7.7 Folds

`split_group_id` is the participant, so all 22 of a participant's device streams
land in one fold.  Assignment is stratified by the release's six condition
groups, ordered inside each group by a keyed digest of seed 20260829, and dealt
round-robin across five folds — no RNG, so two clean roots agree.  P0.2 assigns
outer-fold identity only; train, validation and test labels belong to whatever
milestone actually trains something.

### 7.8 The gate

Sixteen conditions, all required:

```text
P01_AUTHORITY_DEPENDENCY_VERIFIED
ALL_SOURCE_ASSETS_HASH_VERIFIED
PARTICIPANT_INDEX_RECONCILED
ASSESSMENT_INDEX_RECONCILED
STREAM_INDEX_RECONCILED
ALL_SOURCE_SAMPLES_STORED_EXACTLY_ONCE
SOURCE_TIME_TOKENS_PRESERVED
STREAM_ROW_GROUP_INDEX_COMPLETE
SEGMENTS_PARTITION_STREAMS_EXACTLY
WINDOWS_NEVER_CROSS_SEGMENT_BOUNDARIES
WINDOW_SAMPLE_RANGES_REPLAY_EXACTLY
BILATERAL_TASK_PAIRS_COMPLETE
NO_SAMPLE_LEVEL_BILATERAL_SYNC_CLAIM
PARTICIPANT_FOLDS_DISJOINT
INDEPENDENT_MATERIALIZATION_REPRODUCED
NO_VIDEO_SPECTRAL_OR_RESAMPLING_ARTIFACTS
```

Gate status is `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY` or
`NO_GO_PADS_INDEX_AND_WINDOW_MATERIALIZATION`, with
`BLOCKED_P01_DEPENDENCY_UNAVAILABLE` below both.  Exit codes are 0 for an
empirical pass, 3 for a no-go, 4 for a missing authoritative dependency and 2
for an execution error.

Several conditions are checked against something the materializer did not
produce: the sample total against the pinned P0.1 count, replay against the
release's own asset hashes read back from disk, and window ranges against rows
the store actually returned.  Verification is a separate pass over what was
written, not over the arrays that wrote it.

Reproduction reuses the P0.1 receipt split: two child processes materialize
into two empty output roots, and the second is handed the first's receipt.  A
CLI cannot reproduce itself in one process because the receipts would share a
PID — a test asserts exactly that.  As with P0.1 and v0.5D, this establishes two
executions under a trusted procedure and is not remote attestation.

### 7.9 Result

The gate is `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY` on all sixteen conditions,
run against the frozen PhysioNet 1.0.0 release.

| | |
|---|---|
| Participants / assessments / streams | 469 / 5,159 / 10,318 |
| Samples stored | 13,447,168, none duplicated |
| Files re-verified against `SHA256SUMS.txt` | 11,256 |
| Sample store | 41 part files, 10,318 row groups, one per stream, ~910 MB |
| Segments | 14,729 |
| Time gaps above the threshold | **4,411** |
| Four-second windows | 50,676, none crossing a segment |
| Bilateral task pairs | 5,159 (complete) |
| Bilateral window pairs | 23,928 |
| Sample-level alignment claims | 0 |
| Outer folds | 5, sized 96 / 94 / 94 / 93 / 92 |
| Streams replaying byte-exactly | **10,318 / 10,318** |
| Samples read back from the store | **13,447,168** — source = stored = replayed |
| Windows replayed from the store | 50,676, no failures |
| Evidence hash | `7ca16981…`, identical across both processes |
| Storage-index content hash | `22aeeb03…` |

Two numbers deserve reading carefully.

**4,411 gaps.** Fourteen thousand seven hundred and twenty-nine segments over
ten thousand three hundred and eighteen streams means the release contains
4,411 intervals longer than `3 x dt_ref`.  Gap-aware windowing is not a
formality on this corpus: a fixed 4 s slicer anchored on sample counts would
have silently produced windows spanning a discontinuity.

**23,928 bilateral window pairs, against 50,676 windows.** Half of 50,676 is
25,338, so 1,410 windows have no partner: a gap on one wrist removes a window
that still exists on the other.  Those windows stay in the window index and
simply go unpaired.  Inventing a partner would be the sample-level alignment
claim this milestone exists to refuse.

## 8. PADS-P0.3 — source-time spectral preservation and tremor-band replay

### 8.1 The claim

Source-time, gap-aware storage and indexed replay preserve the frequency-domain
content of irregularly sampled PADS wrist signals without interpolation,
resampling or nominal-grid substitution.

That is a storage-and-signal-integrity result. It is not a disease
classification, a tremor-detection accuracy, a video–IMU, a bilateral
sample-fusion or a sampling-rate-ablation result, and the report publishes a
zero count for each artifact those milestones would produce.

### 8.2 The P0.2.1 dependency

P0.3 pins the P0.1 and P0.2.1 evidence hashes, the published P0.2.1 report
bytes, the storage-index content hash, the source manifest and the schema
fingerprints of the two P0.2 tables it reads. It never rebuilds P0.2.

Verification recomputes the storage-index content hash from the index the store
actually holds. A substituted store with the same row counts and a different
content hash is refused — the failure a row-count check cannot see. It also
requires the store's own `_PADS_P02_INDEX_SUCCESS` marker.

### 8.3 The transform

A conventional FFT assumes uniform spacing. PADS does not have it: reference
intervals run from 9.9199 ms to 10.0800 ms across the corpus and individual
deltas from 13.8 µs to 58.8 ms. The kernel therefore evaluates a nonuniform
discrete Fourier transform at the actual sample times, per axis, in this order:
centre time on the window start, fit and remove a linear trend against those
times, apply a continuous-time Hann weight defined on them, transform.

```text
w_i  = 1/2 - 1/2 cos(2*pi*(t_i - t_0)/T)
X(f) = sum_i w_i x_i exp(-j 2*pi*f (t_i - t_0))
P(f) = |X(f)|^2 / sum_i w_i^2
```

The grid is frozen at 3.0–12.0 Hz in 0.25 Hz steps: 37 bins, matching a
four-second window's Rayleigh resolution exactly. No zero-padding and no
oversampled grid, either of which would imply physical resolution this
milestone has not earned. Values are generated from integer millihertz so each
is an exact binary fraction.

Everything is float64. No BLAS call is made — the transform is an elementwise
product and a fixed-shape reduction, not a matrix multiply — so the result does
not depend on threading, and the authoritative run additionally pins
single-threaded numerics and records that it did.

### 8.4 Raw axes, never vector magnitude

Spectra are computed per axis and summed within a sensor family:

```text
P_acc  = P_ax + P_ay + P_az
P_gyro = P_gx + P_gy + P_gz
```

Gyroscope is the primary tremor-frequency workload; accelerometer is
corroborative. Absolute power is never compared between the two — their units
differ.

Vector magnitude is never the primary input, and the kernel controls show why
rather than asserting it: a 5 Hz tone reports 5 Hz on its raw axis and 10 Hz
through `|x|`, because taking the magnitude first doubles the fundamental.

### 8.5 Two frozen sets

The **workload** is one canonical window per stream that has a valid P0.2.1
window — the window whose task-local midpoint lies closest to the stream's own,
ties broken by the earlier start. 358 of the 10,318 streams hold no valid
window, so the set is one per *eligible* stream.

The **audit subset** is stratified by task, wrist, fold, sample count and gap
adjacency, ordered inside each stratum by a keyed SHA-256 digest rather than an
RNG, and capped at ten per populated stratum. A window counts as gap-adjacent
when it is the first window of a segment beginning at a real break or the last
of one ending at a break; a stream merely starting or stopping is not a break.

Both sets are frozen before any spectrum is examined.

### 8.6 The independent source path

For every audited window, the reference spectrum is computed through a path
that does not call the replay API and does not reuse the P0.2 stream reader.
It is a second, minimal implementation of "open the device file the release
published and take the rows this window covers", and it derives the task-local
origin the way the release does — the first row's own `Time` — so it never
consults the store for it.

Using the same spectral kernel on both sides is deliberate: the boundary under
test is source parsing against indexed replay. The kernel itself is validated
separately by twelve synthetic controls, which the audit runs in its own
process rather than deferring to a test suite that may not have been executed.

### 8.7 Eligibility

A window carries a spectrum only when its P0.2.1 status is valid, its stored
times strictly increase, `dt_ref_ps` is present and positive, coverage clears
the P0.2.1 floor, the window lies inside one segment, the grid stays under the
cadence-supported Nyquist limit, and all six channels are finite.

```text
f_Nyquist,reference = 1 / (2 dt_ref)
```

The declared 100 Hz is never used for eligibility when the stored per-stream
`dt_ref_ps` exists, and the sample count is never a gate condition.

### 8.8 The gate

Sixteen conditions, all required:

```text
P02_1_DEPENDENCY_VERIFIED
FREQUENCY_GRID_FROZEN
WORKLOAD_SELECTION_DETERMINISTIC
ONE_CANONICAL_WINDOW_SELECTED_PER_ELIGIBLE_STREAM
SOURCE_TIME_USED_FOR_EVERY_SPECTRUM
NO_NOMINAL_GRID_TIMESTAMP_SUBSTITUTION
DT_REF_USED_FOR_CADENCE_AND_NYQUIST
NO_FIXED_SAMPLE_COUNT_ASSUMPTION
RAW_AXES_PRESERVED
NO_VECTOR_MAGNITUDE_PRIMARY_SIGNAL
SOURCE_AND_REPLAY_ROWS_IDENTICAL
SOURCE_AND_REPLAY_SPECTRA_IDENTICAL
ALL_SPECTRAL_OUTPUT_ROWS_RECONCILED
SYNTHETIC_KERNEL_CONTROLS_PASS
INDEPENDENT_MATERIALIZATION_REPRODUCED
NO_RESAMPLING_RATE_ABLATION_OR_VIDEO_ARTIFACTS
```

Gate status is `PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION` or
`NO_GO_PADS_SPECTRAL_PRESERVATION`, with `BLOCKED_P02_DEPENDENCY_UNAVAILABLE`
below both. The success marker is `_PADS_P03_SPECTRAL_SUCCESS`, never a generic
`_SUCCESS`.

Several conditions are positive probes rather than the absence of a complaint.
The nominal-grid condition rebuilds, for every workload window, the ordinal/rate
grid an implementation might have substituted and requires that the stored
timestamps differ from it — so it cannot pass vacuously. The Nyquist condition
recomputes the limit from each stream's own cadence and reports how many rows
carry the declared-rate 50 Hz value instead; on this corpus none do, because no
stream's `dt_ref` is exactly 10 ms.

### 8.9 Result

The gate is `PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION` on all sixteen
conditions, run against the frozen P0.2.1 store and the original release files.

| | |
|---|---|
| Streams / with a valid window | 10,318 / 9,960 |
| Workload windows | 9,960 over 9,960 distinct streams, all eligible |
| Spectral rows | 19,920 — 9,960 gyroscope, 9,960 accelerometer |
| Independently audited windows | **6,077** across 862 strata |
| Audit coverage | 11 tasks, both wrists, 5 folds, lengths 395–405, 1,497 gap-adjacent + 4,580 interior |
| Row mismatches | 0 |
| Input-hash mismatches | 0 |
| Spectral-hash mismatches | 0 |
| Dominant-frequency mismatches | 0 |
| **Maximum observed bin error** | **0.0** |
| Nominal-grid substitutions | 0, against 9,960 windows that differ from one |
| Nyquist rows from `dt_ref` / from the declared rate | 9,960 / 0 |
| Kernel controls | 12 / 12, run in the audit's own process |
| Evidence hash | `a0be87d4…`, identical across both processes |
| Spectral-table content hash | `27bb6444…` |

The maximum bin error is exactly zero, not a tolerance: both paths feed
identical float64 inputs into the identical kernel, so agreement is bit
equality rather than numerical closeness. If it ever ceases to be zero, that is
a reproducibility incident to diagnose — multithreaded BLAS, platform maths,
serialization metadata or the algorithm — and not an invitation to loosen the
comparison.

Two numbers are worth reading carefully.

**9,960 windows differ from a nominal grid, and 0 were substituted.** The zero
on its own would be satisfied by an implementation that never checked. The
second number is what gives it force: every window's stored timestamps
genuinely differ from the ordinal/rate grid a substitution would have produced.

**The workload carries 9 distinct window lengths; the audit subset carries 11.**
The workload takes one window per stream, so the 395- and 396-sample cases —
which occur once and twice in the entire corpus — are never reached by it. The
stratified audit subset reaches both. Neither set assumes a length.

## 9. PADS-P0.4 — rate ablations and anti-aliasing

### 9.1 The claim

Deriving lower uniform sampling rates from source-time PADS storage preserves
the 3–10 Hz tremor band, and the loss that does appear at 25 Hz is confined to
the 10–12 Hz edge and is reported separately rather than averaged away.

That is a sampling-rate result about a storage system. It is not a disease
classification, a tremor-detection accuracy, a video–IMU or a storage-benchmark
result, and the report publishes a zero count for each artifact those
milestones would produce.

### 9.2 The P0.3 dependency

P0.4 pins the P0.3 evidence hash, the published P0.3 report bytes, the P0.3
spectral-table content hash, the frequency grid, the P0.2.1 chain beneath it —
evidence hash, storage-index content hash, source manifest — and the SHA-256 of
the frozen anti-alias coefficients. It never rebuilds P0.3, and it never
recomputes a native spectrum: the P0.3 table *is* the reference every derived
rate is compared against.

### 9.3 Whole segments, not windows

Rates are derived from whole P0.2.1 contiguous segments. Deriving per window
would let each window invent its own filter state at its own edges, so a
window's spectrum would depend on where the window was cut. Windows are taken
from the derived segment afterwards, and a window is eligible only if it lies
entirely inside the supported output.

### 9.4 Two stages, and why they are separate

**Stage A** brackets the irregular source onto the exact 100 Hz parent grid by
linear interpolation between the two neighbouring source samples. It never
extrapolates: a target with no bracketing pair is not produced.

**Stage B** applies one frozen linear-phase Type I FIR per derived rate,
evaluated through its polyphase branches.

Keeping them separate means exactly one uniformization step exists, and the
anti-alias response that is published is the response that runs. The rejected
alternative — a per-output normalized irregular sinc — would have made the
executed transfer function depend on local sample spacing, so no single
response could have been published at all.

### 9.5 One filter per rate, not one cutoff for all

| Rate | L/M | Passband | Stopband | Taps | Group delay | Edge | Ripple | Stopband |
|---|---|---|---|---|---|---|---|---|
| 50 Hz | 1/2 | 0–12 Hz | 25 Hz | 33 | 16 | +0.0028 dB | 0.0081 dB | 63.80 dB |
| 30 Hz | 3/10 | 0–12 Hz | 15 Hz | 399 | 199 | −0.0034 dB | 0.0093 dB | 64.38 dB |
| 25 Hz | 1/4 | 0–10 Hz | 12.5 Hz | 161 | 80 | −0.0024 dB | 0.0083 dB | 64.57 dB |

A universal fraction-of-Nyquist cutoff was rejected: at 25 Hz it would have
put the transition band across 10 Hz, attenuating the top of the tremor band
that the milestone exists to measure. Each filter is designed against the band
it must preserve. Combined coefficients hash to `976957f7…`.

### 9.6 The branch gains are published, not normalized

Each prototype sums to its upsampling factor, so the effective DC gain is 1.
The three 30 Hz polyphase branches sum to 1.000001881, 0.999996238 and
1.000001881 — a spread of 0.000049 dB, four orders of magnitude inside the
0.25 dB ripple budget.

Those branch gains are left alone. Normalizing each branch separately would
make the executed gain depend on output phase, turning the 3/10 path into a
periodically time-varying correction and replacing one frozen transfer
function with three. The report publishes `prototype_coefficient_sum`,
`upsampling_factor`, `effective_dc_gain`, `polyphase_dc_gains`,
`polyphase_dc_gain_spread_db` and `per_phase_normalization: false` so the
prototype's normalization is never mistaken for the realized gain.

A constant-input control pins the decision empirically: mean gain 1 to within
1e-6, realized per-output ripple equal to the published branch spread, and
gain within a single phase invariant to exactly 0.00e+00. The last number is
the one that matters — it is what would move if anything were being
renormalized per output.

### 9.7 Support is the intersection of both stages

`S_derived = S_100Hz_bracketable ∩ S_FIR_valid`, in that order. An interval the
source could not bracket removes the derived ordinals it feeds *before* the
filter guard is consulted; the filter guard alone would not have removed them.

Nothing is padded, reflected, repeated or renormalized at an edge. An output
whose kernel would run off either end is simply not produced, so support
narrows as the rate falls.

### 9.8 Exact rational time

100, 50 and 25 Hz have exact picosecond periods. 30 Hz does not: its period is
`Fraction(100_000_000_000, 3)` picoseconds. Ordinal *k* is held as exactly
*k*/30 s rather than rounded, and the schema refuses any `_ns` field, which
could not represent the release's ten-decimal times in the first place.

### 9.9 The gate

Eighteen conditions. On an empty fact record the gate satisfies **0 of 18**:
no condition can be met by having measured nothing.

Two are decided by positive probes rather than by the absence of a complaint.

**Support.** Whether a corpus contains an unbracketable interval is a property
of the corpus, so requiring the parent stage to have removed something would
test the corpus instead of the code. A control builds an offset segment
in-process and requires the parent stage to have removed outputs the FIR guard
alone would have admitted — 500, 250, 150 and 125 of them at 100, 50, 30 and
25 Hz. Every derived ordinal in the real run is then re-checked against its own
polyphase kernel support.

**Branch gain.** The realized ripple and the coefficient sums are computed by
different routes, so they agree to 3.9e-15 dB rather than exactly; the gate
allows 1e-9 dB. Per-phase normalization would collapse the realized ripple to
zero — a discrepancy of 4.9e-05 dB, ten million times that bound. Concealment
is what closes the gate, not what slips through it.

### 9.10 Result

The gate is `PASS_PADS_RATE_ABLATION_AND_ANTI_ALIASING` on all eighteen
conditions, run against the frozen P0.3 spectra, the P0.2.1 store and the
original release files.

| | |
|---|---|
| Workload windows | 9,960, from P0.3 unchanged |
| Segment × rate grids | 39,840 |
| Rate-windows attempted / eligible | 39,840 / **38,316** |
| Eligible by rate | 100 Hz 9,960 · 50 Hz 9,756 · 30 Hz 9,327 · 25 Hz 9,273 |
| Derived sample counts | 400 / 200 / 120 / 100 — one value per rate, no exceptions |
| Derived samples written | 7,981,740 |
| Ordinals on exact rational time | 7,981,740 checked, 0 mismatches, 0 rounded at 30 Hz |
| Removed by the parent stage | 730,804 of 25,515,223 the FIR guard alone would have admitted |
| Admitted over an unbracketable parent | **0** |
| Spectral rows | 76,632 — 38,316 gyroscope, 38,316 accelerometer |
| Independently audited windows | 1,689, 5,627 comparisons |
| Derived-value / spectral / sample mismatches | 0 / 0 / 0 |
| **Maximum observed bin error** | **0.0** |
| Participants | 469, 7,504 summary rows — 3,752 core, 3,752 edge |
| Resampling controls | 12 / 12, run in the audit's own process |
| Evidence hash | `2aaebd34…`, identical across both processes |
| Spectral-table content hash | `a66785de…` |

Eligibility falls monotonically with rate — 9,960, 9,756, 9,327, 9,273 —
because a longer kernel reaches further past each segment's ends and those
outputs are refused rather than padded.

#### What the ablation shows

Median across 469 participants, by band:

| Rate | Core 3–10 Hz power | Core distance | Edge 10–12 Hz power | Edge distance | Dominant kept |
|---|---|---|---|---|---|
| 100 Hz | 0.9710 / 0.9726 | 0.0082 / 0.0078 | 0.9126 / 0.9116 | 0.0073 / 0.0090 | 97.5% / 97.9% |
| 50 Hz | 0.9712 / 0.9727 | 0.0082 / 0.0078 | 0.9130 / 0.9121 | 0.0071 / 0.0089 | 97.6% / 97.9% |
| 30 Hz | 0.9712 / 0.9727 | 0.0082 / 0.0077 | 0.9123 / 0.9114 | 0.0074 / 0.0091 | 97.6% / 98.0% |
| 25 Hz | 0.9712 / 0.9728 | 0.0082 / 0.0078 | **0.4401 / 0.4698** | **0.3345 / 0.3057** | 93.9% / 96.1% |

*(accelerometer / gyroscope)*

**The core band does not care about the rate.** 0.9710, 0.9712, 0.9712, 0.9712
— the 3–10 Hz power ratio is flat to four decimals from 100 Hz down to 25 Hz,
and the shape distance is flat at about 0.008. The ~2.9% deficit is already
present at 100 Hz, where no anti-alias filter runs at all, so it is stage A's
uniformization of irregular source time and not a consequence of the rate.

**The 25 Hz edge band is where the cost lands.** Power falls to 0.44/0.47 and
the shape distance rises fortyfold, from 0.008 to 0.31–0.33. That is the
designed behaviour: the 25 Hz filter passes to 10 Hz and stops at 12.5 Hz, so
10–12 Hz sits in its transition band. Separating this band from the core is
the reason the milestone reports two bands rather than one — averaged
together, a real 3–10 Hz preservation result would have been contaminated by a
loss that was deliberately accepted.

**Dominant-frequency preservation drops only at 25 Hz**, from about 97.6% to
93.9% (accelerometer) and 96.1% (gyroscope) — the windows whose peak sat in
the sacrificed edge.

#### A metric that had to be fixed first

The first authoritative run passed all eighteen conditions and published core
power ratios of 0.9738, 0.4869, 0.2922 and 0.2434 — which is rate/100 to five
digits. The P0.3 kernel's power grows linearly with the number of samples
transformed, which never mattered inside P0.3 because every window was only
compared against itself. Here it did: a perfectly preserved signal was being
reported as having lost 75% of its tremor-band power.

The kernel stayed frozen; the normalization was added to P0.4's comparison,
dividing each side by its own sample count. The eighteen conditions were not
wrong to miss it — they ask whether storage and indexed replay preserve what
the source contained, and that answer was bit-exact throughout. A twelfth
control now asks the separate question the gate did not: a 5 Hz tone, inside
every rate's passband, must produce a core ratio near one at all four rates.
Unnormalized it reads 1.00, 0.50, 0.30, 0.25; it now reads 1.000000, 0.999923,
1.000439, 0.999969.

## 10. Milestones after P0.4

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

**PADS-P0.5 — storage and retrieval benchmarks.** Opens only on a passing
P0.4 gate. The same baseline comparison E4D-P0.4 defines, measured against the
PADS source-time store: bytes per hour, index overhead, random four-second
retrieval, sequential replay and deterministic replay hashes. It is a separate
milestone from the four that precede it, and none of them emit its artifacts —
each publishes a zero count for them instead.

## 11. What the paper says about VIDIMU

> Of the evaluated multimodal source structures, reproducibility of a
> publisher-provided transformation was not sufficient to establish temporal
> authority. TremoraStore reproduced all source transformations byte-for-byte
> but withheld aligned materialization because raw polling groups, derived
> motion ordinals, and decoded frames lacked a defensible common mapping.

That is not an embarrassing failure. It is the direct evidence for why the
authority model is necessary.

## 12. Provenance of this branch

The E4D-P0.1 and PADS-P0.1 implementation on this branch is a **re-derivation**.
An earlier build of the same design was produced in an ephemeral environment and
lost with it, together with the adversarial reviews it had passed. This code was
rebuilt from the architecture and has not carried those reviews. It is a new
artifact and the commit messages say so.

## 13. Current status

| Component | Status |
|---|---|
| Cross-dataset timing-authority model | Implemented and frozen; adversarial unit tests only |
| VIDIMU v0.4 Gate B | `PASS`; unchanged |
| VIDIMU v0.5 / v0.5D | `NO-GO`; permanent, unchanged, hash-pinned |
| E4D-P0.1 contract, schemas, parsers, subset, gate | Implemented; not executed against Ego4D assets — the CLI returns `BLOCKED_INPUT_DATA_UNAVAILABLE` |
| E4D-P0.1 empirical gate | Not evaluated; requires the signed Ego4D licence and pinned assets |
| E4D-P0.2 / P0.3 / P0.4 | Closed pending a passing P0.1 gate |
| PADS-P0.1 contract, schemas, parser, audit | Implemented and executed against PADS 1.0.0 |
| PADS-P0.1 empirical gate | `PASS_SOURCE_RELATIVE_UNIMODAL_CLOCK`; 14/14 |
| PADS-P0.2 storage, indexes, windows, folds, replay | Implemented and executed against PADS 1.0.0 |
| PADS-P0.2 empirical gate | `PASS_PADS_INDEX_AND_WINDOW_AUTHORITY`; 16/16, reproduced across two processes |
| PADS-P0.3 spectral preservation | Implemented and executed against the P0.2.1 store |
| PADS-P0.3 empirical gate | `PASS_PADS_SOURCE_TIME_SPECTRAL_PRESERVATION`; 16/16, maximum bin error 0.0, reproduced across two processes |
| PADS-P0.4 rate ablations and anti-aliasing | Implemented and executed against the P0.3 spectra and the P0.2.1 store |
| PADS-P0.4 empirical gate | `PASS_PADS_RATE_ABLATION_AND_ANTI_ALIASING`; 18/18, maximum bin error 0.0, reproduced across two processes |
| PADS-P0.5 | Closed; P0.4 emits no classification, video-association or storage-benchmark result |
| EgoInertia-MI | Verified real (arXiv:2607.03934); optional, off the critical path |
| Comparative systems results | None yet |
