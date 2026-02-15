import gzip
import io
import json
import math
import os
import sqlite3
import threading
from datetime import date
from datetime import datetime, timezone
from typing import Any, Optional
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
    return (
        os.getenv("PITENSOR_DATABASE_URL")
        or os.getenv("DATABASE_PUBLIC_URL")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()


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
