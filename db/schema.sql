-- Idempotent plain-SQL schema for the live recommendation engine.
-- Feedback is the authoritative profile state; no profile vector is persisted.

CREATE EXTENSION IF NOT EXISTS vector;

-- Canonical (year, make, baseModel) recommendation items with the
-- deterministic aggregate features the recommender is built from.
CREATE TABLE IF NOT EXISTS vehicle_items (
    item_id text PRIMARY KEY,
    year integer NOT NULL,
    make text NOT NULL,
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
    variant_count integer NOT NULL CHECK (variant_count > 0)
);

-- Normalized EPA source configurations, keyed by original EPA ID.
CREATE TABLE IF NOT EXISTS vehicle_variants (
    epa_id integer PRIMARY KEY,
    item_id text NOT NULL REFERENCES vehicle_items (item_id),
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

CREATE TABLE IF NOT EXISTS item_embeddings (
    item_id text PRIMARY KEY REFERENCES vehicle_items (item_id),
    embedding vector(32) NOT NULL
);

-- Singleton: catalog provenance plus model bias (NULL until a model is saved).
CREATE TABLE IF NOT EXISTS model_meta (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    snapshot_sha256 text NOT NULL,
    bias double precision
);

-- One-time feedback per canonical item; the complete authoritative history.
CREATE TABLE IF NOT EXISTS feedback (
    item_id text PRIMARY KEY REFERENCES vehicle_items (item_id),
    liked boolean NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now()
);
