# Tremora artifact status

**Status date:** 2026-08-28  
**Authority:** This registry records scope status without modifying historical
artifacts.

| Artifact | Status | SHA-256 |
|---|---|---|
| `TremoraStore_Public_Data_Benchmark_Spec_v0_1.md` | **Current authoritative scope**: public-data-only storage, indexing, synchronization, replay, and regeneration-verified completeness benchmark; VIDIMU v0.4 Gate B is `PASS`, while both the v0.5 RAW-native authority gate and v0.5D source-derived materialization gate are `NO-GO`; v0.6/v0.6D remain closed | `dc8b54ce42792a5b88585a9fee78bde6604befa5898c9c447d785d0d3129cd83` |
| `fullmotion/detector/benchmarks/vidimu_v2_release_audit.json` | **Current implementation evidence**: deterministic `PASS` audit of all 208 pinned VIDIMU CSV/RAW pairs plus central-directory-only original-video candidate inventory | `41277661f9e248da2f42c0703b69beec92bcaf0037b5d46264f64852ab22ecf1` |
| `fullmotion/detector/benchmarks/vidimu_v04_gate_b_release_audit.json` | **Current Gate-B empirical evidence**: `BYTE_IDENTICAL_SOURCE_TO_CV_PASS` / `PASS` for two clean, process/root/inode-disjoint all-208 executions in the frozen observed environment | `d24863d20347cf2c9ab092a9f7771ada3a88ec8fbc77a7b33788df5c0637a10e` |
| `fullmotion/detector/benchmarks/vidimu_v05_sync_authority_audit.json` | **Current v0.5 authority evidence**: audit execution `PASS`, gate `NO_GO_RAW_NATIVE_CLOCK_AUTHORITY`; all 208 original RAW assets and all 217 released synchronization overrides reconcile, two records are `AMBIGUOUS_SOURCE_MAPPING`, and no canonical clocks or v0.6 artifacts were emitted | `3d4492f984ddffaed579da2e107aaf9f7d1e9cdae1ddc83629f8708d8e75bdec` |
| `fullmotion/detector/benchmarks/vidimu_v05d_derived_alignment_release_audit.json` | **Current v0.5D evidence**: audit execution `PASS`, source-derived materialization gate `NO_GO`; all 217 source transformations reproduce byte-for-byte, but 2,036,601 RAW polling groups do not map one-to-one to 299,711 50 Hz STO/MOT ordinals, 30/34 RAW trims end mid group, and no alignment Parquet or success marker was emitted | `131a6110d699ed8d0ebd7611c820112f1fe6af5c0e44f116181bd4a8495ac1b0` |
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
