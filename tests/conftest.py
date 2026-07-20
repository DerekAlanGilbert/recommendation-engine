"""Shared fixtures: an isolated PostgreSQL database for API tests.

Requires the local pgvector database from compose.yaml: `docker compose up -d`.
API tests get a dedicated `rec_test_api` database, created and dropped per
session, so development data and the store-test database are untouched.
"""

from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest

from app.store import database_url

API_TEST_DATABASE = "rec_test_api"


def _database_named(url, name):
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{name}", parts.query, parts.fragment))


@pytest.fixture(scope="session")
def api_database_url():
    admin_url = database_url()
    try:
        with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
            admin.execute(f"DROP DATABASE IF EXISTS {API_TEST_DATABASE} WITH (FORCE)")
            admin.execute(f"CREATE DATABASE {API_TEST_DATABASE}")
    except psycopg.OperationalError as error:
        pytest.fail(f"PostgreSQL unavailable at {admin_url}; run `docker compose up -d` ({error})")
    yield _database_named(admin_url, API_TEST_DATABASE)
    with psycopg.connect(admin_url, autocommit=True, connect_timeout=5) as admin:
        admin.execute(f"DROP DATABASE IF EXISTS {API_TEST_DATABASE} WITH (FORCE)")
