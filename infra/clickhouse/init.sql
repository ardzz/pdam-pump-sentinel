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
