-- Idempotent plain-SQL schema (v2) for the live recommendation engine.
-- Persists the three-level catalog: raw EPA configs, consumer-facing
-- variants (the recommendation targets), and model-year families.
-- Feedback is the authoritative profile state; no posterior is persisted.

CREATE EXTENSION IF NOT EXISTS vector;

-- Legacy v1 detection and explicit reset live in app/store.py. Normal startup
-- fails closed rather than deleting family-level feedback implicitly.

-- (year, make, baseModel) model-year families derived from the raw grouping.
CREATE TABLE IF NOT EXISTS vehicle_families (
    family_id text PRIMARY KEY,
    year integer NOT NULL,
    make text NOT NULL,
    base_model text NOT NULL,
    config_count integer NOT NULL CHECK (config_count > 0)
);

-- (year, make, model) consumer-facing variants — the recommendation targets —
-- with the deterministic aggregate features the recommender is built from.
-- family_id is the variant's primary family (modal baseModel of its configs).
CREATE TABLE IF NOT EXISTS consumer_variants (
    variant_id text PRIMARY KEY,
    family_id text NOT NULL REFERENCES vehicle_families (family_id),
    year integer NOT NULL,
    make text NOT NULL,
    model text NOT NULL,
    base_model text NOT NULL,
    vehicle_class text NOT NULL,
    fuel_type text NOT NULL,
    drive_family text NOT NULL,
    transmission_family text NOT NULL,
    city_mpg double precision NOT NULL,
    highway_mpg double precision NOT NULL,
    combined_mpg double precision NOT NULL,
    cylinders double precision,
    displacement double precision,
    electric_range double precision NOT NULL,
    co2_tailpipe_gpm double precision NOT NULL,
    config_count integer NOT NULL CHECK (config_count > 0)
);

-- Normalized raw EPA configurations, keyed by original EPA ID. family_id is
-- the config's own (year, make, baseModel) family from the raw grouping,
-- which for one known source quirk differs from its variant's primary family.
CREATE TABLE IF NOT EXISTS epa_configs (
    epa_id integer PRIMARY KEY,
    variant_id text NOT NULL REFERENCES consumer_variants (variant_id),
    family_id text NOT NULL REFERENCES vehicle_families (family_id),
    year integer NOT NULL,
    make text NOT NULL,
    model text NOT NULL,
    base_model text NOT NULL,
    vehicle_class text NOT NULL,
    fuel_type text NOT NULL,
    drive text NOT NULL,
    transmission text NOT NULL,
    cylinders integer,
    displacement double precision,
    city_mpg integer NOT NULL,
    highway_mpg integer NOT NULL,
    combined_mpg integer NOT NULL,
    electric_range integer NOT NULL,
    co2_tailpipe_gpm double precision NOT NULL
);

CREATE TABLE IF NOT EXISTS variant_embeddings (
    variant_id text PRIMARY KEY REFERENCES consumer_variants (variant_id),
    embedding vector(32) NOT NULL
);

-- Singleton: catalog provenance plus model bias (NULL until a model is saved).
CREATE TABLE IF NOT EXISTS model_meta (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    snapshot_sha256 text NOT NULL,
    bias double precision
);

-- One-time feedback per consumer variant; the complete authoritative history.
-- event_order preserves a stable audit/replay order even though the current
-- static-likelihood posterior is mathematically order-independent.
CREATE TABLE IF NOT EXISTS feedback (
    variant_id text PRIMARY KEY REFERENCES consumer_variants (variant_id),
    liked boolean NOT NULL,
    event_order bigint GENERATED ALWAYS AS IDENTITY,
    recorded_at timestamptz NOT NULL DEFAULT now()
);
