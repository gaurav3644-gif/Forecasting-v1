import gzip
import io
import json
import math
import os
import sqlite3
import threading
from datetime import date
from datetime import datetime, timezone
from typing import Any, Optional, Dict
from urllib.parse import urlparse

import pandas as pd

_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # numpy/pandas scalars and arrays
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _to_jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(getattr(obj, "tolist")):
        try:
            return _to_jsonable(obj.tolist())
        except Exception:
            pass

    try:
        import pandas as _pd  # type: ignore

        if isinstance(obj, _pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, _pd.Timedelta):
            return float(obj.total_seconds())
        if isinstance(obj, _pd.Series):
            return _to_jsonable(obj.to_dict())
    except Exception:
        pass

    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=True, sort_keys=True)


def _database_url() -> str:
    """
    Postgres connection string.

    Railway typically provides DATABASE_URL automatically when a Postgres plugin is added.
    """
    direct = (
        os.getenv("PITENSOR_DATABASE_URL")
        or os.getenv("DATABASE_PUBLIC_URL")
        or os.getenv("DATABASE_URL")
        # Common provider variants (Render/Supabase/Neon/Railway)
        or os.getenv("POSTGRES_URL")
        or os.getenv("POSTGRESQL_URL")
        or os.getenv("POSTGRES_URL_NON_POOLING")
        or os.getenv("POSTGRESQL_URL_NON_POOLING")
        or ""
    ).strip()
    if direct:
        return direct

    # Fallback: construct from discrete env vars if a platform exposes PG* but not DATABASE_URL.
    host = (os.getenv("PGHOST") or os.getenv("POSTGRES_HOST") or "").strip()
    db = (os.getenv("PGDATABASE") or os.getenv("POSTGRES_DB") or "").strip()
    user = (os.getenv("PGUSER") or os.getenv("POSTGRES_USER") or "").strip()
    pwd = (os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD") or "").strip()
    port = (os.getenv("PGPORT") or os.getenv("POSTGRES_PORT") or "").strip() or "5432"
    if host and db and user:
        # Don't URL-encode here; most providers pass safe credentials in env vars.
        auth = f"{user}:{pwd}@" if pwd else f"{user}@"
        return f"postgresql://{auth}{host}:{port}/{db}"

    return ""


def _use_postgres() -> bool:
    url = _database_url().lower()
    return url.startswith("postgres://") or url.startswith("postgresql://")


def history_backend() -> str:
    return "postgres" if _use_postgres() else "sqlite"


def history_connection_info() -> dict[str, Any]:
    if _use_postgres():
        url = _database_url()
        if url.lower().startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://") :]
        try:
            parsed = urlparse(url)
        except Exception:
            parsed = None
        host = ""
        db = ""
        if parsed is not None:
            host = parsed.hostname or ""
            db = (parsed.path or "").lstrip("/")
        return {
            "backend": "postgres",
            "host": host,
            "database": db,
            "railway_internal": bool(host.endswith(".railway.internal")),
        }
    return {"backend": "sqlite", "path": _db_path()}


def _db_path() -> str:
    # SQLite fallback for local/dev.
    return (os.getenv("PITENSOR_HISTORY_DB") or "pitensor_history.sqlite3").strip()


def _connect() -> sqlite3.Connection:
    # check_same_thread=False because forecast runs in a background thread.
    conn = sqlite3.connect(_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _running_on_railway() -> bool:
    # Railway sets several env vars at runtime; use a loose check.
    return bool(os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_SERVICE_ID"))


def _pg_backend() -> tuple[str, Any, Any]:
    cached = getattr(_pg_backend, "_cached", None)
    if cached is not None:
        return cached

    try:
        import psycopg2  # type: ignore
        import psycopg2.extras  # type: ignore

        cached = ("psycopg2", psycopg2, psycopg2.extras)
        setattr(_pg_backend, "_cached", cached)
        return cached
    except Exception:
        pass

    try:
        import psycopg  # type: ignore
        from psycopg.rows import dict_row  # type: ignore

        cached = ("psycopg3", psycopg, dict_row)
        setattr(_pg_backend, "_cached", cached)
        return cached
    except Exception as e:
        raise RuntimeError(
            "Postgres history storage requires either psycopg2-binary (Python < 3.13) or psycopg[binary] (Python >= 3.13)."
        ) from e


def _pg_connect():
    _kind, driver, _helper = _pg_backend()
    url = _database_url()
    if url.lower().startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://") :]

    host = urlparse(url).hostname or ""
    if host.endswith(".railway.internal") and not _running_on_railway():
        raise RuntimeError(
            "DATABASE_URL points to a Railway internal hostname (not reachable from your laptop). "
            "For local testing, set DATABASE_PUBLIC_URL or PITENSOR_DATABASE_URL to the Railway public/external Postgres URL."
        )
    return driver.connect(url)


def _pg_binary(blob: bytes) -> Any:
    kind, driver, _helper = _pg_backend()
    if kind == "psycopg2":
        return driver.Binary(blob)
    return blob


def _pg_dict_cursor(conn):
    kind, _driver, helper = _pg_backend()
    if kind == "psycopg2":
        return conn.cursor(cursor_factory=helper.RealDictCursor)
    return conn.cursor(row_factory=helper)


def init_db() -> None:
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        email TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS datasets (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        filename TEXT,
                        created_at TEXT NOT NULL,
                        raw_csv_gz BYTEA NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS forecast_runs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        dataset_id INTEGER REFERENCES datasets(id),
                        created_at TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        forecast_csv_gz BYTEA NOT NULL,
                        feature_importance_json TEXT,
                        driver_artifacts_json TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS supply_plans (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id),
                        forecast_run_id INTEGER NOT NULL UNIQUE REFERENCES forecast_runs(id),
                        created_at TEXT NOT NULL,
                        params_json TEXT NOT NULL,
                        supply_export_csv_gz BYTEA NOT NULL,
                        supply_full_csv_gz BYTEA
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS demo_requests (
                        id SERIAL PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        name TEXT,
                        email TEXT NOT NULL,
                        company TEXT,
                        phone TEXT,
                        role TEXT,
                        message TEXT,
                        source_path TEXT,
                        user_email TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS auth_otps (
                        id TEXT PRIMARY KEY,
                        email TEXT NOT NULL,
                        code_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        consumed_at TEXT,
                        ip TEXT,
                        user_agent TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_access (
                        email TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        approved_at TEXT,
                        approved_by TEXT,
                        last_requested_at TEXT,
                        request_count INTEGER NOT NULL DEFAULT 0
                    )
                    """
                )
                # Migrate: add revocation fields if missing (older deployments).
                try:
                    cur.execute("ALTER TABLE user_access ADD COLUMN IF NOT EXISTS revoked_at TEXT")
                    cur.execute("ALTER TABLE user_access ADD COLUMN IF NOT EXISTS revoked_by TEXT")
                    cur.execute("ALTER TABLE user_access ADD COLUMN IF NOT EXISTS revoked_reason TEXT")
                except Exception:
                    # Non-fatal; table may already have columns or DB may be in a transient state.
                    pass
                conn.commit()
                return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT,
                    created_at TEXT NOT NULL,
                    raw_csv_gz BLOB NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    dataset_id INTEGER,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    forecast_csv_gz BLOB NOT NULL,
                    feature_importance_json TEXT,
                    driver_artifacts_json TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(dataset_id) REFERENCES datasets(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS supply_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    forecast_run_id INTEGER NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    supply_export_csv_gz BLOB NOT NULL,
                    supply_full_csv_gz BLOB,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(forecast_run_id) REFERENCES forecast_runs(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS demo_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    name TEXT,
                    email TEXT NOT NULL,
                    company TEXT,
                    phone TEXT,
                    role TEXT,
                    message TEXT,
                    source_path TEXT,
                    user_email TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS auth_otps (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    consumed_at TEXT,
                    ip TEXT,
                    user_agent TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_access (
                    email TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    approved_by TEXT,
                    last_requested_at TEXT,
                    request_count INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            # Migrate: add revocation fields if missing.
            try:
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(user_access)").fetchall()}
                if "revoked_at" not in cols:
                    conn.execute("ALTER TABLE user_access ADD COLUMN revoked_at TEXT")
                if "revoked_by" not in cols:
                    conn.execute("ALTER TABLE user_access ADD COLUMN revoked_by TEXT")
                if "revoked_reason" not in cols:
                    conn.execute("ALTER TABLE user_access ADD COLUMN revoked_reason TEXT")
            except Exception:
                pass
            conn.commit()
        finally:
            conn.close()


def _gzip_bytes(text: str) -> bytes:
    return gzip.compress((text or "").encode("utf-8"))


def _gunzip_text(blob: bytes) -> str:
    return gzip.decompress(blob).decode("utf-8")


def _bytea_to_bytes(blob: Any) -> Optional[bytes]:
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        return bytes(blob)
    if isinstance(blob, memoryview):
        return blob.tobytes()
    return bytes(blob)


def _df_to_csv_gz(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return _gzip_bytes(buf.getvalue())


def _csv_gz_to_df(blob: bytes) -> pd.DataFrame:
    text = _gunzip_text(blob)
    if not text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(text))


def _get_or_create_user_id(conn: sqlite3.Connection, email: str) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise ValueError("email is required")
    row = conn.execute("SELECT id FROM users WHERE email = ?", (email_norm,)).fetchone()
    if row:
        return int(row["id"])
    conn.execute("INSERT INTO users(email, created_at) VALUES(?, ?)", (email_norm, _utc_now_iso()))
    return int(conn.execute("SELECT id FROM users WHERE email = ?", (email_norm,)).fetchone()["id"])


def _get_or_create_user_id_pg(conn, email: str) -> int:
    email_norm = (email or "").strip().lower()
    if not email_norm:
        raise ValueError("email is required")
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO users(email, created_at)
        VALUES(%s, %s)
        ON CONFLICT(email) DO UPDATE SET email=excluded.email
        RETURNING id
        """,
        (email_norm, _utc_now_iso()),
    )
    row = cur.fetchone()
    return int(row[0])


def save_dataset(email: str, filename: str, raw_df: pd.DataFrame) -> int:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                raw_blob = _df_to_csv_gz(raw_df)
                created_at = _utc_now_iso()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO datasets(user_id, filename, created_at, raw_csv_gz) VALUES(%s, %s, %s, %s) RETURNING id",
                    (user_id, filename or None, created_at, _pg_binary(raw_blob)),
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return int(new_id)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            raw_blob = _df_to_csv_gz(raw_df)
            created_at = _utc_now_iso()
            conn.execute(
                "INSERT INTO datasets(user_id, filename, created_at, raw_csv_gz) VALUES(?, ?, ?, ?)",
                (user_id, filename or None, created_at, raw_blob),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return int(row["id"])
        finally:
            conn.close()


def save_forecast_run(
    email: str,
    dataset_id: Optional[int],
    params: dict[str, Any],
    forecast_df: pd.DataFrame,
    feature_importance: Optional[dict[str, Any]] = None,
    driver_artifacts: Optional[dict[str, Any]] = None,
) -> int:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                created_at = _utc_now_iso()
                params_json = _json_dumps(params or {})
                forecast_blob = _df_to_csv_gz(forecast_df)
                fi_json = _json_dumps(feature_importance or {})
                da_json = _json_dumps(driver_artifacts or {})
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO forecast_runs(
                        user_id, dataset_id, created_at, params_json, forecast_csv_gz, feature_importance_json, driver_artifacts_json
                    )
                    VALUES(%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        user_id,
                        int(dataset_id) if dataset_id is not None else None,
                        created_at,
                        params_json,
                        _pg_binary(forecast_blob),
                        fi_json,
                        da_json,
                    ),
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return int(new_id)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            created_at = _utc_now_iso()
            params_json = _json_dumps(params or {})
            forecast_blob = _df_to_csv_gz(forecast_df)
            fi_json = _json_dumps(feature_importance or {})
            da_json = _json_dumps(driver_artifacts or {})
            conn.execute(
                """
                INSERT INTO forecast_runs(
                    user_id, dataset_id, created_at, params_json, forecast_csv_gz, feature_importance_json, driver_artifacts_json
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, dataset_id, created_at, params_json, forecast_blob, fi_json, da_json),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return int(row["id"])
        finally:
            conn.close()


def save_supply_plan(
    email: str,
    forecast_run_id: int,
    params: dict[str, Any],
    supply_export_df: pd.DataFrame,
    supply_full_df: Optional[pd.DataFrame] = None,
) -> int:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                created_at = _utc_now_iso()
                params_json = _json_dumps(params or {})
                export_blob = _df_to_csv_gz(supply_export_df)
                full_blob = _df_to_csv_gz(supply_full_df) if isinstance(supply_full_df, pd.DataFrame) else None
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO supply_plans(
                        user_id, forecast_run_id, created_at, params_json, supply_export_csv_gz, supply_full_csv_gz
                    )
                    VALUES(%s, %s, %s, %s, %s, %s)
                    ON CONFLICT(forecast_run_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        params_json=excluded.params_json,
                        supply_export_csv_gz=excluded.supply_export_csv_gz,
                        supply_full_csv_gz=excluded.supply_full_csv_gz
                    RETURNING id
                    """,
                    (
                        user_id,
                        int(forecast_run_id),
                        created_at,
                        params_json,
                        _pg_binary(export_blob),
                        _pg_binary(full_blob) if full_blob is not None else None,
                    ),
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return int(new_id)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            created_at = _utc_now_iso()
            params_json = _json_dumps(params or {})
            export_blob = _df_to_csv_gz(supply_export_df)
            full_blob = _df_to_csv_gz(supply_full_df) if isinstance(supply_full_df, pd.DataFrame) else None
            # One supply plan per forecast_run_id (latest wins).
            conn.execute(
                """
                INSERT INTO supply_plans(
                    user_id, forecast_run_id, created_at, params_json, supply_export_csv_gz, supply_full_csv_gz
                )
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(forecast_run_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    params_json=excluded.params_json,
                    supply_export_csv_gz=excluded.supply_export_csv_gz,
                    supply_full_csv_gz=excluded.supply_full_csv_gz
                """,
                (user_id, int(forecast_run_id), created_at, params_json, export_blob, full_blob),
            )
            conn.commit()
            row = conn.execute("SELECT id FROM supply_plans WHERE forecast_run_id = ?", (int(forecast_run_id),)).fetchone()
            return int(row["id"])
        finally:
            conn.close()


def list_forecast_runs(email: str, limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        fr.id AS run_id,
                        fr.created_at AS created_at,
                        fr.params_json AS params_json,
                        d.filename AS filename,
                        CASE WHEN sp.id IS NULL THEN 0 ELSE 1 END AS has_supply_plan
                    FROM forecast_runs fr
                    LEFT JOIN datasets d ON d.id = fr.dataset_id
                    LEFT JOIN supply_plans sp ON sp.forecast_run_id = fr.id
                    WHERE fr.user_id = %s
                    ORDER BY fr.id DESC
                    LIMIT %s
                    """,
                    (user_id, int(limit)),
                )
                rows = cur.fetchall() or []
                out: list[dict[str, Any]] = []
                for r in rows:
                    params = {}
                    try:
                        params = json.loads(r.get("params_json") or "{}")
                    except Exception:
                        params = {}
                    has_supply_plan_val = int(r.get("has_supply_plan") or 0)
                    out.append(
                        {
                            "run_id": int(r.get("run_id")),
                            "created_at": r.get("created_at"),
                            "filename": r.get("filename"),
                            "start_month": params.get("start_month"),
                            "months": params.get("months"),
                            "grain": params.get("grain"),
                            "has_supply_plan": bool(has_supply_plan_val),
                        }
                    )
                return out
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            rows = conn.execute(
                """
                SELECT
                    fr.id AS run_id,
                    fr.created_at AS created_at,
                    fr.params_json AS params_json,
                    d.filename AS filename,
                    CASE WHEN sp.id IS NULL THEN 0 ELSE 1 END AS has_supply_plan
                FROM forecast_runs fr
                LEFT JOIN datasets d ON d.id = fr.dataset_id
                LEFT JOIN supply_plans sp ON sp.forecast_run_id = fr.id
                WHERE fr.user_id = ?
                ORDER BY fr.id DESC
                LIMIT ?
                """,
                (user_id, int(limit)),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                params = {}
                try:
                    params = json.loads(r["params_json"] or "{}")
                except Exception:
                    params = {}
                out.append(
                    {
                        "run_id": int(r["run_id"]),
                        "created_at": r["created_at"],
                        "filename": r["filename"],
                        "start_month": params.get("start_month"),
                        "months": params.get("months"),
                        "grain": params.get("grain"),
                        "has_supply_plan": bool(int(r["has_supply_plan"] or 0)),
                    }
                )
            return out
        finally:
            conn.close()


def list_forecast_runs_admin(*, limit: int = 50) -> list[dict[str, Any]]:
    """
    Admin-only: list runs across all users.

    NOTE: Callers must enforce authorization (e.g., check PITENSOR_ADMIN_EMAILS).
    """
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        fr.id AS run_id,
                        fr.created_at AS created_at,
                        fr.params_json AS params_json,
                        d.filename AS filename,
                        u.email AS user_email,
                        CASE WHEN sp.id IS NULL THEN 0 ELSE 1 END AS has_supply_plan
                    FROM forecast_runs fr
                    INNER JOIN users u ON u.id = fr.user_id
                    LEFT JOIN datasets d ON d.id = fr.dataset_id
                    LEFT JOIN supply_plans sp ON sp.forecast_run_id = fr.id
                    ORDER BY fr.id DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall() or []
                out: list[dict[str, Any]] = []
                for r in rows:
                    params = {}
                    try:
                        params = json.loads(r.get("params_json") or "{}")
                    except Exception:
                        params = {}
                    has_supply_plan_val = int(r.get("has_supply_plan") or 0)
                    out.append(
                        {
                            "run_id": int(r.get("run_id")),
                            "created_at": r.get("created_at"),
                            "filename": r.get("filename"),
                            "user_email": r.get("user_email"),
                            "start_month": params.get("start_month"),
                            "months": params.get("months"),
                            "grain": params.get("grain"),
                            "has_supply_plan": bool(has_supply_plan_val),
                        }
                    )
                return out
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT
                    fr.id AS run_id,
                    fr.created_at AS created_at,
                    fr.params_json AS params_json,
                    d.filename AS filename,
                    u.email AS user_email,
                    CASE WHEN sp.id IS NULL THEN 0 ELSE 1 END AS has_supply_plan
                FROM forecast_runs fr
                INNER JOIN users u ON u.id = fr.user_id
                LEFT JOIN datasets d ON d.id = fr.dataset_id
                LEFT JOIN supply_plans sp ON sp.forecast_run_id = fr.id
                ORDER BY fr.id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                params = {}
                try:
                    params = json.loads(r["params_json"] or "{}")
                except Exception:
                    params = {}
                out.append(
                    {
                        "run_id": int(r["run_id"]),
                        "created_at": r["created_at"],
                        "filename": r["filename"],
                        "user_email": r["user_email"],
                        "start_month": params.get("start_month"),
                        "months": params.get("months"),
                        "grain": params.get("grain"),
                        "has_supply_plan": bool(int(r["has_supply_plan"] or 0)),
                    }
                )
            return out
        finally:
            conn.close()


def load_forecast_run(email: str, run_id: int) -> dict[str, Any]:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        fr.id AS id,
                        fr.created_at AS created_at,
                        fr.params_json AS params_json,
                        fr.forecast_csv_gz AS forecast_csv_gz,
                        fr.feature_importance_json AS feature_importance_json,
                        fr.driver_artifacts_json AS driver_artifacts_json,
                        d.raw_csv_gz AS raw_csv_gz,
                        d.filename AS filename
                    FROM forecast_runs fr
                    LEFT JOIN datasets d ON d.id = fr.dataset_id
                    WHERE fr.user_id = %s AND fr.id = %s
                    """,
                    (user_id, int(run_id)),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError("forecast run not found")
                params = json.loads(row.get("params_json") or "{}")
                fi = json.loads(row.get("feature_importance_json") or "{}")
                da = json.loads(row.get("driver_artifacts_json") or "{}")
                raw_blob = _bytea_to_bytes(row.get("raw_csv_gz"))
                forecast_blob = _bytea_to_bytes(row.get("forecast_csv_gz"))
                raw_df = _csv_gz_to_df(raw_blob) if raw_blob is not None else pd.DataFrame()
                forecast_df = _csv_gz_to_df(forecast_blob or b"")
                return {
                    "run_id": int(row.get("id")),
                    "created_at": row.get("created_at"),
                    "filename": row.get("filename"),
                    "params": params,
                    "raw_df": raw_df,
                    "forecast_df": forecast_df,
                    "feature_importance": fi,
                    "driver_artifacts": da,
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            row = conn.execute(
                """
                SELECT fr.*, d.raw_csv_gz AS raw_csv_gz, d.filename AS filename
                FROM forecast_runs fr
                LEFT JOIN datasets d ON d.id = fr.dataset_id
                WHERE fr.user_id = ? AND fr.id = ?
                """,
                (user_id, int(run_id)),
            ).fetchone()
            if not row:
                raise KeyError("forecast run not found")
            params = json.loads(row["params_json"] or "{}")
            fi = json.loads(row["feature_importance_json"] or "{}")
            da = json.loads(row["driver_artifacts_json"] or "{}")
            raw_df = _csv_gz_to_df(row["raw_csv_gz"]) if row["raw_csv_gz"] is not None else pd.DataFrame()
            forecast_df = _csv_gz_to_df(row["forecast_csv_gz"])
            return {
                "run_id": int(row["id"]),
                "created_at": row["created_at"],
                "filename": row["filename"],
                "params": params,
                "raw_df": raw_df,
                "forecast_df": forecast_df,
                "feature_importance": fi,
                "driver_artifacts": da,
            }
        finally:
            conn.close()


def load_forecast_run_admin(*, run_id: int) -> dict[str, Any]:
    """
    Admin-only: load a forecast run by id regardless of owner.

    NOTE: Callers must enforce authorization (e.g., check PITENSOR_ADMIN_EMAILS).
    """
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        fr.id AS id,
                        fr.created_at AS created_at,
                        fr.params_json AS params_json,
                        fr.forecast_csv_gz AS forecast_csv_gz,
                        fr.feature_importance_json AS feature_importance_json,
                        fr.driver_artifacts_json AS driver_artifacts_json,
                        d.raw_csv_gz AS raw_csv_gz,
                        d.filename AS filename,
                        u.email AS user_email
                    FROM forecast_runs fr
                    INNER JOIN users u ON u.id = fr.user_id
                    LEFT JOIN datasets d ON d.id = fr.dataset_id
                    WHERE fr.id = %s
                    """,
                    (int(run_id),),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError("forecast run not found")
                params = json.loads(row.get("params_json") or "{}")
                fi = json.loads(row.get("feature_importance_json") or "{}")
                da = json.loads(row.get("driver_artifacts_json") or "{}")
                raw_blob = _bytea_to_bytes(row.get("raw_csv_gz"))
                forecast_blob = _bytea_to_bytes(row.get("forecast_csv_gz"))
                raw_df = _csv_gz_to_df(raw_blob) if raw_blob is not None else pd.DataFrame()
                forecast_df = _csv_gz_to_df(forecast_blob or b"")
                return {
                    "run_id": int(row.get("id")),
                    "created_at": row.get("created_at"),
                    "filename": row.get("filename"),
                    "user_email": row.get("user_email"),
                    "params": params,
                    "raw_df": raw_df,
                    "forecast_df": forecast_df,
                    "feature_importance": fi,
                    "driver_artifacts": da,
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT fr.*, d.raw_csv_gz AS raw_csv_gz, d.filename AS filename, u.email AS user_email
                FROM forecast_runs fr
                INNER JOIN users u ON u.id = fr.user_id
                LEFT JOIN datasets d ON d.id = fr.dataset_id
                WHERE fr.id = ?
                """,
                (int(run_id),),
            ).fetchone()
            if not row:
                raise KeyError("forecast run not found")
            params = json.loads(row["params_json"] or "{}")
            fi = json.loads(row["feature_importance_json"] or "{}")
            da = json.loads(row["driver_artifacts_json"] or "{}")
            raw_df = _csv_gz_to_df(row["raw_csv_gz"]) if row["raw_csv_gz"] is not None else pd.DataFrame()
            forecast_df = _csv_gz_to_df(row["forecast_csv_gz"])
            return {
                "run_id": int(row["id"]),
                "created_at": row["created_at"],
                "filename": row["filename"],
                "user_email": row["user_email"],
                "params": params,
                "raw_df": raw_df,
                "forecast_df": forecast_df,
                "feature_importance": fi,
                "driver_artifacts": da,
            }
        finally:
            conn.close()


def load_supply_plan(email: str, run_id: int) -> dict[str, Any]:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        sp.forecast_run_id AS forecast_run_id,
                        sp.created_at AS created_at,
                        sp.params_json AS params_json,
                        sp.supply_export_csv_gz AS supply_export_csv_gz,
                        sp.supply_full_csv_gz AS supply_full_csv_gz
                    FROM supply_plans sp
                    INNER JOIN forecast_runs fr ON fr.id = sp.forecast_run_id
                    WHERE sp.user_id = %s AND fr.id = %s
                    """,
                    (user_id, int(run_id)),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError("supply plan not found")
                params = json.loads(row.get("params_json") or "{}")
                export_blob = _bytea_to_bytes(row.get("supply_export_csv_gz"))
                full_blob = _bytea_to_bytes(row.get("supply_full_csv_gz"))
                export_df = _csv_gz_to_df(export_blob or b"")
                full_df = _csv_gz_to_df(full_blob) if full_blob is not None else pd.DataFrame()
                return {
                    "forecast_run_id": int(row.get("forecast_run_id")),
                    "created_at": row.get("created_at"),
                    "params": params,
                    "supply_plan_df": export_df,
                    "supply_plan_full_df": full_df,
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            row = conn.execute(
                """
                SELECT sp.*
                FROM supply_plans sp
                INNER JOIN forecast_runs fr ON fr.id = sp.forecast_run_id
                WHERE sp.user_id = ? AND fr.id = ?
                """,
                (user_id, int(run_id)),
            ).fetchone()
            if not row:
                raise KeyError("supply plan not found")
            params = json.loads(row["params_json"] or "{}")
            export_df = _csv_gz_to_df(row["supply_export_csv_gz"])
            full_df = _csv_gz_to_df(row["supply_full_csv_gz"]) if row["supply_full_csv_gz"] is not None else pd.DataFrame()
            return {
                "forecast_run_id": int(row["forecast_run_id"]),
                "created_at": row["created_at"],
                "params": params,
                "supply_plan_df": export_df,
                "supply_plan_full_df": full_df,
            }
        finally:
            conn.close()


def load_supply_plan_admin(*, run_id: int) -> dict[str, Any]:
    """
    Admin-only: load a supply plan by forecast_run_id regardless of owner.

    NOTE: Callers must enforce authorization (e.g., check PITENSOR_ADMIN_EMAILS).
    """
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT
                        sp.forecast_run_id AS forecast_run_id,
                        sp.created_at AS created_at,
                        sp.params_json AS params_json,
                        sp.supply_export_csv_gz AS supply_export_csv_gz,
                        sp.supply_full_csv_gz AS supply_full_csv_gz,
                        u.email AS user_email
                    FROM supply_plans sp
                    INNER JOIN forecast_runs fr ON fr.id = sp.forecast_run_id
                    INNER JOIN users u ON u.id = fr.user_id
                    WHERE fr.id = %s
                    """,
                    (int(run_id),),
                )
                row = cur.fetchone()
                if not row:
                    raise KeyError("supply plan not found")
                params = json.loads(row.get("params_json") or "{}")
                export_blob = _bytea_to_bytes(row.get("supply_export_csv_gz"))
                full_blob = _bytea_to_bytes(row.get("supply_full_csv_gz"))
                export_df = _csv_gz_to_df(export_blob or b"")
                full_df = _csv_gz_to_df(full_blob) if full_blob is not None else pd.DataFrame()
                return {
                    "forecast_run_id": int(row.get("forecast_run_id")),
                    "created_at": row.get("created_at"),
                    "user_email": row.get("user_email"),
                    "params": params,
                    "supply_plan_df": export_df,
                    "supply_plan_full_df": full_df,
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute(
                """
                SELECT sp.*, u.email AS user_email
                FROM supply_plans sp
                INNER JOIN forecast_runs fr ON fr.id = sp.forecast_run_id
                INNER JOIN users u ON u.id = fr.user_id
                WHERE fr.id = ?
                """,
                (int(run_id),),
            ).fetchone()
            if not row:
                raise KeyError("supply plan not found")
            params = json.loads(row["params_json"] or "{}")
            export_df = _csv_gz_to_df(row["supply_export_csv_gz"])
            full_df = _csv_gz_to_df(row["supply_full_csv_gz"]) if row["supply_full_csv_gz"] is not None else pd.DataFrame()
            return {
                "forecast_run_id": int(row["forecast_run_id"]),
                "created_at": row["created_at"],
                "user_email": row["user_email"],
                "params": params,
                "supply_plan_df": export_df,
                "supply_plan_full_df": full_df,
            }
        finally:
            conn.close()

def save_demo_request(
    *,
    name: str,
    email: str,
    company: Optional[str] = None,
    phone: Optional[str] = None,
    role: Optional[str] = None,
    message: Optional[str] = None,
    source_path: Optional[str] = None,
    user_email: Optional[str] = None,
) -> int:
    init_db()
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("valid email is required")
    company = (company or "").strip() or None
    phone = (phone or "").strip() or None
    role = (role or "").strip() or None
    message = (message or "").strip() or None
    source_path = (source_path or "").strip() or None
    user_email = (user_email or "").strip().lower() or None
    created_at = _utc_now_iso()

    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO demo_requests(
                        created_at, name, email, company, phone, role, message, source_path, user_email
                    )
                    VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (created_at, name or None, email, company, phone, role, message, source_path, user_email),
                )
                new_id = cur.fetchone()[0]
                conn.commit()
                return int(new_id)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO demo_requests(
                    created_at, name, email, company, phone, role, message, source_path, user_email
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, name or None, email, company, phone, role, message, source_path, user_email),
            )
            conn.commit()
            row = conn.execute("SELECT last_insert_rowid() AS id").fetchone()
            return int(row["id"])
        finally:
            conn.close()


def delete_forecast_run(email: str, run_id: int) -> bool:
    """
    Delete a forecast run (and its supply plan) for the given user.

    Returns True if something was deleted, False if the run was not found for that user.
    """
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                user_id = _get_or_create_user_id_pg(conn, email)
                cur = conn.cursor()
                cur.execute(
                    "SELECT dataset_id FROM forecast_runs WHERE id = %s AND user_id = %s",
                    (int(run_id), int(user_id)),
                )
                row = cur.fetchone()
                if not row:
                    return False
                dataset_id = row[0]

                cur.execute(
                    "DELETE FROM supply_plans WHERE forecast_run_id = %s AND user_id = %s",
                    (int(run_id), int(user_id)),
                )
                cur.execute(
                    "DELETE FROM forecast_runs WHERE id = %s AND user_id = %s",
                    (int(run_id), int(user_id)),
                )

                if dataset_id is not None:
                    cur.execute("SELECT 1 FROM forecast_runs WHERE dataset_id = %s LIMIT 1", (int(dataset_id),))
                    still_used = cur.fetchone() is not None
                    if not still_used:
                        cur.execute(
                            "DELETE FROM datasets WHERE id = %s AND user_id = %s",
                            (int(dataset_id), int(user_id)),
                        )

                conn.commit()
                return True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            user_id = _get_or_create_user_id(conn, email)
            row = conn.execute(
                "SELECT dataset_id FROM forecast_runs WHERE id = ? AND user_id = ?",
                (int(run_id), int(user_id)),
            ).fetchone()
            if not row:
                return False
            dataset_id = row["dataset_id"]

            conn.execute(
                "DELETE FROM supply_plans WHERE forecast_run_id = ? AND user_id = ?",
                (int(run_id), int(user_id)),
            )
            conn.execute(
                "DELETE FROM forecast_runs WHERE id = ? AND user_id = ?",
                (int(run_id), int(user_id)),
            )

            if dataset_id is not None:
                still_used = (
                    conn.execute(
                        "SELECT 1 FROM forecast_runs WHERE dataset_id = ? LIMIT 1",
                        (int(dataset_id),),
                    ).fetchone()
                    is not None
                )
                if not still_used:
                    conn.execute(
                        "DELETE FROM datasets WHERE id = ? AND user_id = ?",
                        (int(dataset_id), int(user_id)),
                    )

            conn.commit()
            return True
        finally:
            conn.close()


def delete_forecast_run_admin(*, run_id: int) -> bool:
    """
    Admin-only: delete a forecast run (and its supply plan) regardless of owner.

    Returns True if deleted, False if not found.
    """
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT dataset_id, user_id FROM forecast_runs WHERE id = %s", (int(run_id),))
                row = cur.fetchone()
                if not row:
                    return False
                dataset_id, user_id = row[0], row[1]

                cur.execute("DELETE FROM supply_plans WHERE forecast_run_id = %s", (int(run_id),))
                cur.execute("DELETE FROM forecast_runs WHERE id = %s", (int(run_id),))

                if dataset_id is not None:
                    cur.execute("SELECT 1 FROM forecast_runs WHERE dataset_id = %s LIMIT 1", (int(dataset_id),))
                    still_used = cur.fetchone() is not None
                    if not still_used:
                        cur.execute(
                            "DELETE FROM datasets WHERE id = %s AND user_id = %s",
                            (int(dataset_id), int(user_id)),
                        )

                conn.commit()
                return True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute(
                "SELECT dataset_id, user_id FROM forecast_runs WHERE id = ?",
                (int(run_id),),
            ).fetchone()
            if not row:
                return False
            dataset_id, user_id = row["dataset_id"], row["user_id"]

            conn.execute("DELETE FROM supply_plans WHERE forecast_run_id = ?", (int(run_id),))
            conn.execute("DELETE FROM forecast_runs WHERE id = ?", (int(run_id),))

            if dataset_id is not None:
                still_used = (
                    conn.execute(
                        "SELECT 1 FROM forecast_runs WHERE dataset_id = ? LIMIT 1",
                        (int(dataset_id),),
                    ).fetchone()
                    is not None
                )
                if not still_used:
                    conn.execute(
                        "DELETE FROM datasets WHERE id = ? AND user_id = ?",
                        (int(dataset_id), int(user_id)),
                    )

            conn.commit()
            return True
        finally:
            conn.close()

def create_auth_otp(
    *,
    otp_id: str,
    email: str,
    code_hash: str,
    expires_at: str,
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> None:
    """
    Persist a one-time sign-in code for OTP-based authentication.
    """
    init_db()
    otp_id = (otp_id or "").strip()
    email = (email or "").strip().lower()
    code_hash = (code_hash or "").strip()
    if not otp_id:
        raise ValueError("otp_id is required")
    if "@" not in email:
        raise ValueError("valid email is required")
    if not code_hash:
        raise ValueError("code_hash is required")
    created_at = _utc_now_iso()
    expires_at = (expires_at or "").strip() or created_at
    ip = (ip or "").strip() or None
    user_agent = (user_agent or "").strip() or None

    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO auth_otps(
                        id, email, code_hash, created_at, expires_at, attempts, consumed_at, ip, user_agent
                    )
                    VALUES(%s, %s, %s, %s, %s, 0, NULL, %s, %s)
                    """,
                    (otp_id, email, code_hash, created_at, expires_at, ip, user_agent),
                )
                conn.commit()
                return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO auth_otps(
                    id, email, code_hash, created_at, expires_at, attempts, consumed_at, ip, user_agent
                )
                VALUES(?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (otp_id, email, code_hash, created_at, expires_at, ip, user_agent),
            )
            conn.commit()
        finally:
            conn.close()


def get_auth_otp(*, otp_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    otp_id = (otp_id or "").strip()
    if not otp_id:
        return None

    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM auth_otps WHERE id = %s", (otp_id,))
                row = cur.fetchone()
                if not row:
                    return None
                # psycopg row is a dict-like via _pg_row_helper
                if isinstance(row, dict):
                    return dict(row)
                # fallback: positional (shouldn't happen with our cursor helper)
                return {
                    "id": row[0],
                    "email": row[1],
                    "code_hash": row[2],
                    "created_at": row[3],
                    "expires_at": row[4],
                    "attempts": row[5],
                    "consumed_at": row[6],
                    "ip": row[7],
                    "user_agent": row[8],
                }
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM auth_otps WHERE id = ?", (otp_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def increment_auth_otp_attempts(*, otp_id: str) -> None:
    init_db()
    otp_id = (otp_id or "").strip()
    if not otp_id:
        return
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("UPDATE auth_otps SET attempts = attempts + 1 WHERE id = %s", (otp_id,))
                conn.commit()
                return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        conn = _connect()
        try:
            conn.execute("UPDATE auth_otps SET attempts = attempts + 1 WHERE id = ?", (otp_id,))
            conn.commit()
        finally:
            conn.close()


def consume_auth_otp(*, otp_id: str) -> None:
    init_db()
    otp_id = (otp_id or "").strip()
    if not otp_id:
        return
    consumed_at = _utc_now_iso()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("UPDATE auth_otps SET consumed_at = %s WHERE id = %s", (consumed_at, otp_id))
                conn.commit()
                return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        conn = _connect()
        try:
            conn.execute("UPDATE auth_otps SET consumed_at = ? WHERE id = ?", (consumed_at, otp_id))
            conn.commit()
        finally:
            conn.close()


def get_latest_auth_otp_created_at(*, email: str) -> Optional[str]:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        return None
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT created_at FROM auth_otps WHERE email = %s ORDER BY created_at DESC LIMIT 1",
                    (email,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                if isinstance(row, dict):
                    return row.get("created_at")
                return row[0]
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT created_at FROM auth_otps WHERE email = ? ORDER BY created_at DESC LIMIT 1",
                (email,),
            ).fetchone()
            if not row:
                return None
            return row["created_at"]
        finally:
            conn.close()


def get_user_access(*, email: str) -> Optional[Dict[str, Any]]:
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        return None
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute("SELECT * FROM user_access WHERE email = %s", (email,))
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute("SELECT * FROM user_access WHERE email = ?", (email,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def upsert_user_access_request(*, email: str) -> None:
    """
    Ensure a user_access record exists and bump request_count + last_requested_at.
    """
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("valid email is required")
    now = _utc_now_iso()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count)
                    VALUES(%s, %s, NULL, NULL, %s, 1)
                    ON CONFLICT(email) DO UPDATE SET
                        last_requested_at=excluded.last_requested_at,
                        request_count=user_access.request_count + 1
                    """,
                    (email, now, now),
                )
                conn.commit()
                return
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count)
                VALUES(?, ?, NULL, NULL, ?, 1)
                ON CONFLICT(email) DO UPDATE SET
                    last_requested_at=excluded.last_requested_at,
                    request_count=request_count + 1
                """,
                (email, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def approve_user_access(*, email: str, approved_by: Optional[str] = None) -> bool:
    init_db()
    email = (email or "").strip().lower()
    approved_by = (approved_by or "").strip().lower() or None
    if "@" not in email:
        raise ValueError("valid email is required")
    now = _utc_now_iso()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count)
                    VALUES(%s, %s, %s, %s, NULL, 0)
                    ON CONFLICT(email) DO UPDATE SET
                        approved_at=excluded.approved_at,
                        approved_by=excluded.approved_by,
                        revoked_at=NULL,
                        revoked_by=NULL,
                        revoked_reason=NULL
                    """,
                    (email, now, now, approved_by),
                )
                conn.commit()
                return True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count)
                VALUES(?, ?, ?, ?, NULL, 0)
                ON CONFLICT(email) DO UPDATE SET
                    approved_at=excluded.approved_at,
                    approved_by=excluded.approved_by,
                    revoked_at=NULL,
                    revoked_by=NULL,
                    revoked_reason=NULL
                """,
                (email, now, now, approved_by),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def is_user_access_approved(*, email: str) -> bool:
    rec = get_user_access(email=email)
    if not rec:
        return False
    if (rec.get("revoked_at") or "").strip():
        return False
    return bool((rec.get("approved_at") or "").strip())


def is_user_access_revoked(*, email: str) -> bool:
    rec = get_user_access(email=email)
    if not rec:
        return False
    return bool((rec.get("revoked_at") or "").strip())


def revoke_user_access(*, email: str, revoked_by: Optional[str] = None, reason: Optional[str] = None) -> bool:
    init_db()
    email = (email or "").strip().lower()
    revoked_by = (revoked_by or "").strip().lower() or None
    reason = (reason or "").strip() or None
    if "@" not in email:
        raise ValueError("valid email is required")
    now = _utc_now_iso()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count, revoked_at, revoked_by, revoked_reason)
                    VALUES(%s, %s, NULL, NULL, NULL, 0, %s, %s, %s)
                    ON CONFLICT(email) DO UPDATE SET
                        approved_at=NULL,
                        approved_by=NULL,
                        revoked_at=excluded.revoked_at,
                        revoked_by=excluded.revoked_by,
                        revoked_reason=excluded.revoked_reason
                    """,
                    (email, now, now, revoked_by, reason),
                )
                conn.commit()
                return True
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO user_access(email, created_at, approved_at, approved_by, last_requested_at, request_count, revoked_at, revoked_by, revoked_reason)
                VALUES(?, ?, NULL, NULL, NULL, 0, ?, ?, ?)
                ON CONFLICT(email) DO UPDATE SET
                    approved_at=NULL,
                    approved_by=NULL,
                    revoked_at=excluded.revoked_at,
                    revoked_by=excluded.revoked_by,
                    revoked_reason=excluded.revoked_reason
                """,
                (email, now, now, revoked_by, reason),
            )
            conn.commit()
            return True
        finally:
            conn.close()


def list_user_access(*, limit: int = 500, status: str = "all") -> list[dict[str, Any]]:
    """
    Admin utility: list access records.

    status: all|pending|approved|revoked
    """
    init_db()
    status = (status or "all").strip().lower()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                where = ""
                if status == "pending":
                    where = "WHERE approved_at IS NULL AND (revoked_at IS NULL OR revoked_at = '')"
                elif status == "approved":
                    where = "WHERE approved_at IS NOT NULL AND (revoked_at IS NULL OR revoked_at = '')"
                elif status == "revoked":
                    where = "WHERE revoked_at IS NOT NULL AND revoked_at <> ''"
                cur.execute(
                    f"""
                    SELECT email, created_at, approved_at, approved_by, last_requested_at, request_count, revoked_at, revoked_by, revoked_reason
                    FROM user_access
                    {where}
                    ORDER BY COALESCE(last_requested_at, created_at) DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            where = ""
            if status == "pending":
                where = "WHERE approved_at IS NULL AND (revoked_at IS NULL OR revoked_at = '')"
            elif status == "approved":
                where = "WHERE approved_at IS NOT NULL AND (revoked_at IS NULL OR revoked_at = '')"
            elif status == "revoked":
                where = "WHERE revoked_at IS NOT NULL AND revoked_at <> ''"
            rows = conn.execute(
                f"""
                SELECT email, created_at, approved_at, approved_by, last_requested_at, request_count, revoked_at, revoked_by, revoked_reason
                FROM user_access
                {where}
                ORDER BY COALESCE(last_requested_at, created_at) DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def list_current_users_admin(*, limit: int = 500) -> list[dict[str, Any]]:
    """
    Admin utility: list current (non-revoked) users.

    Includes:
      - Approved users in user_access
      - "Legacy" users present in users table but missing user_access

    Excludes:
      - Revoked users
      - Pending users (user_access exists but approved_at is NULL)
    """
    init_db()
    out: list[dict[str, Any]] = []
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    WITH access AS (
                        SELECT
                            email,
                            approved_at,
                            approved_by,
                            revoked_at,
                            revoked_by,
                            revoked_reason,
                            last_requested_at,
                            request_count,
                            created_at AS access_created_at
                        FROM user_access
                    ),
                    base AS (
                        SELECT
                            u.email AS email,
                            u.created_at AS user_created_at,
                            a.access_created_at AS access_created_at,
                            a.approved_at AS approved_at
                        FROM users u
                        LEFT JOIN access a ON a.email = u.email
                        WHERE (a.revoked_at IS NULL OR a.revoked_at = '')
                          AND (a.email IS NULL OR a.approved_at IS NOT NULL)
                        UNION ALL
                        SELECT
                            a.email AS email,
                            NULL AS user_created_at,
                            a.access_created_at AS access_created_at,
                            a.approved_at AS approved_at
                        FROM access a
                        WHERE a.approved_at IS NOT NULL
                          AND (a.revoked_at IS NULL OR a.revoked_at = '')
                          AND NOT EXISTS (SELECT 1 FROM users u WHERE u.email = a.email)
                    )
                    SELECT email, user_created_at, access_created_at, approved_at
                    FROM base
                    ORDER BY COALESCE(user_created_at, approved_at, access_created_at) DESC NULLS LAST
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall() or []
                seen: set[str] = set()
                for r in rows:
                    email = str(r.get("email") or "").strip().lower()
                    if not email or email in seen:
                        continue
                    seen.add(email)
                    approved_at = r.get("approved_at")
                    out.append(
                        {
                            "email": email,
                            "approved_at": approved_at,
                            "created_at": r.get("user_created_at"),
                            "status": "approved" if (approved_at or "").strip() else "legacy",
                        }
                    )
                return out
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            users_rows = conn.execute("SELECT email, created_at FROM users ORDER BY created_at DESC").fetchall()
            access_rows = conn.execute(
                """
                SELECT email, approved_at, revoked_at, created_at
                FROM user_access
                """
            ).fetchall()
            access_map = {str(r["email"]).strip().lower(): dict(r) for r in access_rows if r and r.get("email")}

            seen: set[str] = set()
            # Users table first (most relevant; they have at least one run history)
            for r in users_rows:
                email = str(r["email"]).strip().lower()
                if not email or email in seen:
                    continue
                a = access_map.get(email) or {}
                if (a.get("revoked_at") or "").strip():
                    continue
                if a and not (a.get("approved_at") or "").strip():
                    # Pending: exclude from current list (shows in Pending Users)
                    continue
                seen.add(email)
                approved_at = (a.get("approved_at") or "").strip() or None
                out.append(
                    {
                        "email": email,
                        "approved_at": approved_at,
                        "created_at": r.get("created_at"),
                        "status": "approved" if approved_at else "legacy",
                    }
                )

            # Add approved access records that don't have any runs yet (not in users table)
            for email, a in access_map.items():
                if email in seen:
                    continue
                if (a.get("revoked_at") or "").strip():
                    continue
                if not (a.get("approved_at") or "").strip():
                    continue
                seen.add(email)
                out.append(
                    {
                        "email": email,
                        "approved_at": (a.get("approved_at") or "").strip() or None,
                        "created_at": None,
                        "status": "approved",
                    }
                )

            return out[: int(limit)]
        finally:
            conn.close()

def list_pending_user_access(*, limit: int = 200) -> list[dict[str, Any]]:
    init_db()
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = _pg_dict_cursor(conn)
                cur.execute(
                    """
                    SELECT email, created_at, approved_at, approved_by, last_requested_at, request_count
                    FROM user_access
                    WHERE approved_at IS NULL AND (revoked_at IS NULL OR revoked_at = '')
                    ORDER BY last_requested_at DESC NULLS LAST, created_at DESC
                    LIMIT %s
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            rows = conn.execute(
                """
                SELECT email, created_at, approved_at, approved_by, last_requested_at, request_count
                FROM user_access
                WHERE approved_at IS NULL AND (revoked_at IS NULL OR revoked_at = '')
                ORDER BY COALESCE(last_requested_at, created_at) DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def delete_user_admin(*, email: str) -> bool:
    """
    Admin-only: delete a user and all their stored history (runs, datasets, supply plans) + access record.
    Returns True if user existed in either users or user_access and deletion was attempted.
    """
    init_db()
    email = (email or "").strip().lower()
    if "@" not in email:
        raise ValueError("valid email is required")
    existed = False
    with _LOCK:
        if _use_postgres():
            conn = _pg_connect()
            try:
                cur = conn.cursor()
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                row = cur.fetchone()
                user_id = row[0] if row else None
                if user_id is not None:
                    existed = True
                    # Children first
                    cur.execute("DELETE FROM supply_plans WHERE user_id = %s", (int(user_id),))
                    cur.execute("DELETE FROM forecast_runs WHERE user_id = %s", (int(user_id),))
                    cur.execute("DELETE FROM datasets WHERE user_id = %s", (int(user_id),))
                    cur.execute("DELETE FROM users WHERE id = %s", (int(user_id),))
                # Access + OTPs are email-keyed
                cur.execute("DELETE FROM auth_otps WHERE email = %s", (email,))
                cur.execute("DELETE FROM user_access WHERE email = %s", (email,))
                if cur.rowcount:
                    existed = True
                conn.commit()
                return existed
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        conn = _connect()
        try:
            row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            user_id = row["id"] if row else None
            if user_id is not None:
                existed = True
                conn.execute("DELETE FROM supply_plans WHERE user_id = ?", (int(user_id),))
                conn.execute("DELETE FROM forecast_runs WHERE user_id = ?", (int(user_id),))
                conn.execute("DELETE FROM datasets WHERE user_id = ?", (int(user_id),))
                conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
            conn.execute("DELETE FROM auth_otps WHERE email = ?", (email,))
            cur = conn.execute("DELETE FROM user_access WHERE email = ?", (email,))
            if getattr(cur, "rowcount", 0):
                existed = True
            conn.commit()
            return existed
        finally:
            conn.close()
