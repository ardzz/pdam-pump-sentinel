CREATE TABLE IF NOT EXISTS telemetry_observations
(
    observed_at DateTime64(3, 'UTC'),
    ingested_at DateTime64(3, 'UTC'),
    device_id String,
    measurement LowCardinality(String),
    value_float Nullable(Float64),
    value_int Nullable(Int64),
    value_string Nullable(String),
    value_bool Nullable(UInt8),
    unit Nullable(String),
    quality Nullable(String),
    tags Map(String, String),
    attributes Map(String, String),
    metadata Map(String, String)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(observed_at)
ORDER BY (device_id, measurement, observed_at);

CREATE TABLE IF NOT EXISTS operator_labels
(
    station LowCardinality(String),
    source_timestamp String,
    label UInt8,
    label_source LowCardinality(String),
    operator_id Nullable(String),
    reason Nullable(String),
    created_at DateTime64(3, 'UTC'),
    updated_at DateTime64(3, 'UTC')
)
ENGINE = ReplacingMergeTree(updated_at)
PARTITION BY toYYYYMM(created_at)
ORDER BY (station, source_timestamp, label_source);
