"""Explicit schemas for the four tables P0.5 publishes."""

from __future__ import annotations

import pyarrow as pa

from .contract import P05_CONTRACT_VERSION, P05_SCHEMA_VERSION

_METADATA = {
    b"tremora.schema_version": P05_SCHEMA_VERSION.encode(),
    b"tremora.contract_version": P05_CONTRACT_VERSION.encode(),
    b"tremora.time_unit": b"NANOSECOND",
}


def _schema(fields: list[tuple[str, pa.DataType]], table: str) -> pa.Schema:
    return pa.schema(
        [pa.field(name, kind, nullable=False) for name, kind in fields],
        metadata={**_METADATA, b"tremora.table": table.encode()},
    )


def pads_p05_storage_schema() -> pa.Schema:
    return _schema([
        ("representation", pa.string()),
        ("source_payload_bytes", pa.int64()),
        ("physical_storage_bytes", pa.int64()),
        ("metadata_bytes", pa.int64()),
        ("index_bytes", pa.int64()),
        ("unique_samples", pa.int64()),
        ("stored_sample_instances", pa.int64()),
        ("duplicate_sample_instances", pa.int64()),
        ("duplication_factor", pa.float64()),
        ("bytes_per_unique_sample", pa.float64()),
        ("bytes_per_stream", pa.float64()),
        ("bytes_per_window", pa.float64()),
        ("compression_ratio_vs_original_source", pa.float64()),
        ("file_count", pa.int64()),
        ("compression_codec", pa.string()),
        ("compression_level", pa.string()),
    ], "pads_p05_storage")


def pads_p05_retrieval_schema() -> pa.Schema:
    return _schema([
        ("representation", pa.string()),
        ("query_class", pa.string()),
        ("query_id", pa.string()),
        ("round_id", pa.int32()),
        ("latency_ns", pa.int64()),
        ("cpu_time_ns", pa.int64()),
        ("rows_returned", pa.int64()),
        ("bytes_returned", pa.int64()),
        ("peak_rss_delta", pa.int64()),
        ("content_sha256", pa.string()),
        ("status", pa.string()),
    ], "pads_p05_retrieval")


def pads_p05_latency_summary_schema() -> pa.Schema:
    return _schema([
        ("representation", pa.string()),
        ("query_class", pa.string()),
        ("queries", pa.int64()),
        ("rows_returned", pa.int64()),
        ("mean_latency_ns", pa.float64()),
        ("p50_latency_ns", pa.float64()),
        ("p95_latency_ns", pa.float64()),
        ("p99_latency_ns", pa.float64()),
        ("cpu_time_ns_total", pa.int64()),
        ("peak_rss_delta_max", pa.int64()),
        ("rows_per_second", pa.float64()),
        ("queries_per_second", pa.float64()),
        ("measured_rounds", pa.int32()),
    ], "pads_p05_latency_summary")


def pads_p05_participant_latency_schema() -> pa.Schema:
    return _schema([
        ("participant_id", pa.string()),
        ("condition_group", pa.string()),
        ("representation", pa.string()),
        ("query_class", pa.string()),
        ("queries", pa.int64()),
        ("median_latency_ns", pa.float64()),
        ("p95_latency_ns", pa.float64()),
    ], "pads_p05_participant_latency")


P05_TABLE_SCHEMAS = {
    "pads_p05_storage": pads_p05_storage_schema,
    "pads_p05_retrieval": pads_p05_retrieval_schema,
    "pads_p05_latency_summary": pads_p05_latency_summary_schema,
    "pads_p05_participant_latency": pads_p05_participant_latency_schema,
}

P05_TABLE_FILES = {
    "pads_p05_storage": "pads_p05_storage.parquet",
    "pads_p05_retrieval": "pads_p05_retrieval.parquet",
    "pads_p05_latency_summary": "pads_p05_latency_summary.parquet",
    "pads_p05_participant_latency": "pads_p05_participant_latency.parquet",
}


__all__ = [
    "P05_TABLE_FILES",
    "P05_TABLE_SCHEMAS",
    "pads_p05_latency_summary_schema",
    "pads_p05_participant_latency_schema",
    "pads_p05_retrieval_schema",
    "pads_p05_storage_schema",
]
