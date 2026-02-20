"""
metrics_store.py — SQLite persistence for Conductor model metrics.

Saves metrics after benchmarks so Conductor starts warm after a restart.
Includes staleness checks and reliability decay so transient failures
don't permanently penalize a model.
"""

import sqlite3
import time
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "conductor_metrics.db")

# Data older than this is ignored on load — triggers fresh benchmark instead
STALE_AFTER_HOURS = 24

# Reliability decays back toward this neutral baseline over time
RELIABILITY_NEUTRAL = 0.8
# Time (hours) to fully recover from a bad reliability score
RELIABILITY_RECOVERY_HOURS = 12


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist yet."""
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                model_name      TEXT PRIMARY KEY,
                latency         REAL,
                cost            REAL,
                reliability     REAL,
                input_tokens    INTEGER,
                output_tokens   INTEGER,
                tokens_per_sec  REAL,
                error_rate      REAL,
                total_calls     INTEGER,
                total_errors    INTEGER,
                saved_at        REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conductor_state (
                key     TEXT PRIMARY KEY,
                value   REAL
            )
        """)
        conn.commit()


def save_metrics(model_name: str, metrics: dict):
    """Save a single model's metrics after a benchmark run."""
    with _connect() as conn:
        conn.execute("""
            INSERT INTO model_metrics
                (model_name, latency, cost, reliability, input_tokens, output_tokens,
                 tokens_per_sec, error_rate, total_calls, total_errors, saved_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_name) DO UPDATE SET
                latency        = excluded.latency,
                cost           = excluded.cost,
                reliability    = excluded.reliability,
                input_tokens   = excluded.input_tokens,
                output_tokens  = excluded.output_tokens,
                tokens_per_sec = excluded.tokens_per_sec,
                error_rate     = excluded.error_rate,
                total_calls    = excluded.total_calls,
                total_errors   = excluded.total_errors,
                saved_at       = excluded.saved_at
        """, (
            model_name,
            metrics.get("latency"),
            metrics.get("cost"),
            metrics.get("reliability"),
            metrics.get("input_tokens"),
            metrics.get("output_tokens"),
            metrics.get("tokens_per_sec"),
            metrics.get("error_rate", 0.0),
            metrics.get("total_calls", 0),
            metrics.get("total_errors", 0),
            time.time(),
        ))
        conn.commit()


def load_all_metrics() -> dict:
    """
    Load persisted metrics for warm startup.

    Returns {} (empty) if data is stale — caller should run fresh benchmarks.

    For fresh data, applies reliability decay: a model that was struggling
    when last saved gets its reliability nudged back toward neutral (0.8)
    proportional to how much time has passed. This prevents a transient
    failure from permanently haunting a model across restarts.
    """
    stale_cutoff = time.time() - (STALE_AFTER_HOURS * 3600)

    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM model_metrics WHERE saved_at > ?", (stale_cutoff,)
        ).fetchall()

    if not rows:
        return {}

    now = time.time()
    result = {}
    for row in rows:
        age_hours = (now - row["saved_at"]) / 3600
        decay_factor = min(age_hours / RELIABILITY_RECOVERY_HOURS, 1.0)

        saved_reliability = row["reliability"] or RELIABILITY_NEUTRAL
        # Pull reliability toward neutral based on age
        recovered_reliability = saved_reliability + (RELIABILITY_NEUTRAL - saved_reliability) * decay_factor

        result[row["model_name"]] = {
            "latency":        row["latency"],
            "cost":           row["cost"],
            "reliability":    round(recovered_reliability, 3),
            "input_tokens":   row["input_tokens"],
            "output_tokens":  row["output_tokens"],
            "tokens_per_sec": row["tokens_per_sec"],
            "error_rate":     row["error_rate"],
            "total_calls":    row["total_calls"],
            "total_errors":   row["total_errors"],
        }

    return result


def save_benchmark_time(t: float):
    with _connect() as conn:
        conn.execute("""
            INSERT INTO conductor_state (key, value) VALUES ('last_benchmark_time', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (t,))
        conn.commit()


def load_benchmark_time() -> float:
    with _connect() as conn:
        row = conn.execute(
            "SELECT value FROM conductor_state WHERE key = 'last_benchmark_time'"
        ).fetchone()
    return row["value"] if row else 0.0
