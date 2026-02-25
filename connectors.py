import time
import logging
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter()

# ── Connector schemas ──────────────────────────────────────────────────────────

def get_connector_schema(connector: str):
    connector = connector.lower()
    schemas = {
        "database": {
            "fields": [
                {"name": "provider",  "type": "select",   "label": "Provider",
                 "options": ["PostgreSQL", "MySQL", "SQL Server", "Oracle", "Other"], "default": "PostgreSQL"},
                {"name": "host",      "type": "text",     "label": "Host"},
                {"name": "port",      "type": "text",     "label": "Port"},
                {"name": "username",  "type": "text",     "label": "Username"},
                {"name": "password",  "type": "password", "label": "Password"},
                {"name": "database",  "type": "text",     "label": "Database"},
            ]
        },
        "warehouse": {
            "fields": [
                {"name": "provider",    "type": "select",   "label": "Provider",
                 "options": ["Snowflake", "BigQuery", "Redshift", "Azure Synapse", "Other"], "default": "Snowflake"},
                {"name": "account",     "type": "text",     "label": "Account / Project"},
                {"name": "host",        "type": "text",     "label": "Host"},
                {"name": "port",        "type": "text",     "label": "Port"},
                {"name": "username",    "type": "text",     "label": "Username"},
                {"name": "password",    "type": "password", "label": "Password / Token"},
                {"name": "database",    "type": "text",     "label": "Database"},
                {"name": "schema",      "type": "text",     "label": "Schema"},
            ]
        },
        "ecommerce": {
            "fields": [
                {"name": "provider",    "type": "select",   "label": "Provider",
                 "options": ["Shopify", "WooCommerce", "Magento", "BigCommerce", "Other"], "default": "Shopify"},
                {"name": "store_url",   "type": "text",     "label": "Store URL"},
                {"name": "api_key",     "type": "password", "label": "API Key / Token"},
                {"name": "api_secret",  "type": "password", "label": "API Secret (if applicable)"},
                {"name": "start_date",  "type": "date",     "label": "Start Date"},
                {"name": "end_date",    "type": "date",     "label": "End Date"},
            ]
        },
        "pos": {
            "fields": [
                {"name": "provider",      "type": "select",   "label": "POS Provider",
                 "options": ["Square", "Toast", "Clover", "Lightspeed", "Other"], "default": "Square"},
                {"name": "api_base_url",  "type": "text",     "label": "API Base URL"},
                {"name": "api_key",       "type": "password", "label": "API Key / Token"},
                {"name": "location_id",   "type": "text",     "label": "Location / Store ID"},
                {"name": "start_date",    "type": "date",     "label": "Start Date"},
                {"name": "end_date",      "type": "date",     "label": "End Date"},
            ]
        },
        "erp": {
            "fields": [
                {"name": "provider",       "type": "select",   "label": "ERP Provider",
                 "options": ["SAP", "Oracle ERP", "Microsoft Dynamics", "Other"], "default": "SAP"},
                {"name": "api_base_url",   "type": "text",     "label": "API Base URL"},
                {"name": "client_id",      "type": "text",     "label": "Client ID / Username"},
                {"name": "client_secret",  "type": "password", "label": "Client Secret / Password"},
                {"name": "tenant",         "type": "text",     "label": "Tenant / Company ID"},
                {"name": "start_date",     "type": "date",     "label": "Start Date"},
                {"name": "end_date",       "type": "date",     "label": "End Date"},
            ]
        },
    }
    return schemas.get(connector)


@router.get("/connector-schema/{connector}")
def connector_schema(connector: str):
    schema = get_connector_schema(connector)
    if not schema:
        return JSONResponse({"error": "Unknown connector"}, status_code=400)
    return JSONResponse(schema)


# ── Database helpers ───────────────────────────────────────────────────────────

def _db_connect_external(provider: str, creds: dict):
    """Open a connection to the user-supplied external database."""
    host = str(creds.get("host") or "").strip()
    raw_port = creds.get("port") or ""
    try:
        port = int(str(raw_port).strip()) if str(raw_port).strip() else None
    except (ValueError, TypeError):
        port = None
    user = str(creds.get("username") or creds.get("user") or "").strip()
    pwd  = str(creds.get("password") or "").strip()
    db   = str(creds.get("database") or "").strip()

    if provider == "PostgreSQL":
        # Try psycopg2 first, then psycopg3
        try:
            import psycopg2  # type: ignore
            return psycopg2.connect(
                host=host, port=port or 5432,
                user=user, password=pwd, dbname=db,
                connect_timeout=10,
            )
        except ImportError:
            pass
        try:
            import psycopg  # type: ignore
            return psycopg.connect(
                host=host, port=port or 5432,
                user=user, password=pwd, dbname=db,
                connect_timeout=10,
            )
        except ImportError:
            raise RuntimeError(
                "No PostgreSQL driver available. "
                "psycopg2-binary or psycopg[binary] must be installed."
            )

    if provider == "MySQL":
        try:
            import pymysql  # type: ignore
            return pymysql.connect(
                host=host, port=port or 3306,
                user=user, password=pwd, database=db,
                connect_timeout=10,
            )
        except ImportError:
            raise RuntimeError("PyMySQL is not installed. Contact support to enable MySQL imports.")

    raise NotImplementedError(
        f"Direct import is not yet supported for '{provider}'. "
        "Supported providers: PostgreSQL, MySQL."
    )


def _list_tables_external(conn, provider: str) -> list:
    """Return a list of user-accessible table names."""
    cur = conn.cursor()
    if provider in ("PostgreSQL",):
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
    elif provider == "MySQL":
        cur.execute("SHOW TABLES")
    else:
        cur.execute(
            "SELECT table_name FROM information_schema.tables ORDER BY table_name"
        )
    return [row[0] for row in cur.fetchall()]


# ── /connect/{connector} ───────────────────────────────────────────────────────

@router.post("/connect/{connector}")
async def connect(connector: str, request: Request):
    payload = {}
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}

    # Database: actually test connection and return table list
    if connector == "database":
        provider = str(payload.get("provider") or "PostgreSQL").strip()
        try:
            conn = _db_connect_external(provider, payload)
            tables = _list_tables_external(conn, provider)
            try:
                conn.close()
            except Exception:
                pass
            return JSONResponse({
                "status": "success",
                "connector": connector,
                "tables": tables,
            })
        except NotImplementedError as e:
            return JSONResponse({"status": "info", "message": str(e), "connector": connector})
        except Exception as e:
            logging.warning(f"[CONNECTOR] DB connect failed ({provider}): {e}")
            return JSONResponse({"status": "error", "message": str(e), "connector": connector})

    # Other connectors: save config to session (stub — real API integrations go here)
    try:
        email = getattr(request.state, "user_email", None)
        email_norm = (str(email).strip().lower() if email else "")
        session_id = f"user:{email_norm}" if email_norm else "default"
        store = getattr(request.app.state, "data_store", None)
        if isinstance(store, dict):
            s = store.setdefault(session_id, {})
            cfgs = s.setdefault("connector_configs", {})
            cfgs[str(connector).lower()] = {"ts": time.time(), "payload": payload}
    except Exception:
        pass

    return JSONResponse({"status": "success", "connector": connector})


# ── /connector/import ──────────────────────────────────────────────────────────

@router.post("/connector/import")
async def connector_import(request: Request):
    """
    Fetch data from a connected external database table and save it as a dataset
    so it appears in the user's Data page.
    """
    import pandas as pd

    payload = {}
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        pass

    connector   = str(payload.get("connector") or "database").strip()
    table_name  = str(payload.get("table") or "").strip()
    credentials = payload.get("credentials") or {}
    row_limit   = int(payload.get("row_limit") or 200_000)

    if not table_name:
        return JSONResponse({"status": "error", "message": "No table selected."})

    email = getattr(request.state, "user_email", None)
    if not email:
        return JSONResponse({"status": "error", "message": "Not authenticated."})

    if connector == "database":
        provider = str(credentials.get("provider") or "PostgreSQL").strip()
        try:
            conn = _db_connect_external(provider, credentials)
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Connection failed: {e}"})

        try:
            # Safe table name quoting for PostgreSQL/MySQL
            safe_table = table_name.replace('"', '').replace("'", "")
            df = pd.read_sql(f'SELECT * FROM "{safe_table}" LIMIT {row_limit}', conn)
        except Exception as e:
            try:
                conn.close()
            except Exception:
                pass
            return JSONResponse({"status": "error", "message": f"Failed to fetch table '{table_name}': {e}"})

        try:
            conn.close()
        except Exception:
            pass

        if df is None or df.empty:
            return JSONResponse({"status": "error", "message": f"Table '{table_name}' is empty."})

        try:
            import history_store
            filename = f"{table_name} ({provider})"
            dataset_id = history_store.save_dataset(str(email), filename, df)
            logging.info(f"[CONNECTOR] Imported table '{table_name}' ({len(df)} rows) for {email} → dataset_id={dataset_id}")
            return JSONResponse({
                "status": "success",
                "dataset_id": dataset_id,
                "filename": filename,
                "rows": len(df),
                "columns": list(df.columns),
            })
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Failed to save dataset: {e}"})

    return JSONResponse({
        "status": "error",
        "message": f"Import is not yet supported for connector type '{connector}'.",
    })


@router.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    return JSONResponse({"filename": file.filename, "status": "uploaded"})


@router.get("/quickstart-csv")
def quickstart_csv():
    return RedirectResponse(url="/quickstart-csv-section")
