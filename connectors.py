import time
from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse

router = APIRouter()

# Example connector schemas
def get_connector_schema(connector: str):
    connector = connector.lower()
    schemas = {
        "mysql": {
            "fields": [
                {"name": "host", "type": "text", "label": "Host"},
                {"name": "port", "type": "number", "label": "Port", "default": 3306},
                {"name": "username", "type": "text", "label": "Username"},
                {"name": "password", "type": "password", "label": "Password"},
                {"name": "database", "type": "text", "label": "Database"}
            ]
        },
        "postgresql": {
            "fields": [
                {"name": "host", "type": "text", "label": "Host"},
                {"name": "port", "type": "number", "label": "Port", "default": 5432},
                {"name": "username", "type": "text", "label": "Username"},
                {"name": "password", "type": "password", "label": "Password"},
                {"name": "database", "type": "text", "label": "Database"}
            ]
        },
        "mssql": {
            "fields": [
                {"name": "host", "type": "text", "label": "Host"},
                {"name": "port", "type": "number", "label": "Port", "default": 1433},
                {"name": "username", "type": "text", "label": "Username"},
                {"name": "password", "type": "password", "label": "Password"},
                {"name": "database", "type": "text", "label": "Database"}
            ]
        },
        "database": {
            "fields": [
                {"name": "provider", "type": "select", "label": "Provider", "options": ["MySQL", "PostgreSQL", "SQL Server", "Oracle", "Other"], "default": "MySQL"},
                {"name": "host", "type": "text", "label": "Host"},
                {"name": "port", "type": "number", "label": "Port"},
                {"name": "username", "type": "text", "label": "Username"},
                {"name": "password", "type": "password", "label": "Password"},
                {"name": "database", "type": "text", "label": "Database"}
            ]
        },
        "google_sheets": {
            "fields": [
                {"name": "sheet_url", "type": "text", "label": "Google Sheet URL"},
                {"name": "api_key", "type": "text", "label": "API Key"}
            ]
        },
        "shopify": {
            "fields": [
                {"name": "shop_url", "type": "text", "label": "Shopify Store URL"},
                {"name": "api_key", "type": "text", "label": "API Key"},
                {"name": "api_password", "type": "password", "label": "API Password"}
            ]
        },
        "warehouse": {
            "fields": [
                {"name": "provider", "type": "select", "label": "Provider", "options": ["Snowflake", "BigQuery", "Redshift", "Azure Synapse", "Other"], "default": "Snowflake"},
                {"name": "account", "type": "text", "label": "Account / Project"},
                {"name": "host", "type": "text", "label": "Host"},
                {"name": "port", "type": "number", "label": "Port"},
                {"name": "username", "type": "text", "label": "Username"},
                {"name": "password", "type": "password", "label": "Password / Token"},
                {"name": "database", "type": "text", "label": "Database"},
                {"name": "schema", "type": "text", "label": "Schema"},
            ]
        },
        "ecommerce": {
            "fields": [
                {"name": "provider", "type": "select", "label": "Provider", "options": ["Shopify", "WooCommerce", "Magento", "BigCommerce", "Other"], "default": "Shopify"},
                {"name": "store_url", "type": "text", "label": "Store URL"},
                {"name": "api_key", "type": "password", "label": "API Key / Token"},
                {"name": "api_secret", "type": "password", "label": "API Secret (if applicable)"},
                {"name": "start_date", "type": "date", "label": "Start Date"},
                {"name": "end_date", "type": "date", "label": "End Date"},
            ]
        },
        "pos": {
            "fields": [
                {"name": "provider", "type": "select", "label": "POS Provider", "options": ["Square", "Toast", "Clover", "Lightspeed", "Other"], "default": "Square"},
                {"name": "api_base_url", "type": "text", "label": "API Base URL"},
                {"name": "api_key", "type": "password", "label": "API Key / Token"},
                {"name": "location_id", "type": "text", "label": "Location / Store ID"},
                {"name": "start_date", "type": "date", "label": "Start Date"},
                {"name": "end_date", "type": "date", "label": "End Date"},
            ]
        },
        "erp": {
            "fields": [
                {"name": "provider", "type": "select", "label": "ERP Provider", "options": ["SAP", "Oracle ERP", "Microsoft Dynamics", "Other"], "default": "SAP"},
                {"name": "api_base_url", "type": "text", "label": "API Base URL"},
                {"name": "client_id", "type": "text", "label": "Client ID / Username"},
                {"name": "client_secret", "type": "password", "label": "Client Secret / Password"},
                {"name": "tenant", "type": "text", "label": "Tenant / Company ID"},
                {"name": "start_date", "type": "date", "label": "Start Date"},
                {"name": "end_date", "type": "date", "label": "End Date"},
            ]
        }
    }
    return schemas.get(connector)

@router.get("/connector-schema/{connector}")
def connector_schema(connector: str):
    schema = get_connector_schema(connector)
    if not schema:
        return JSONResponse({"error": "Unknown connector"}, status_code=400)
    return JSONResponse(schema)

@router.post("/connect/{connector}")
async def connect(connector: str, request: Request):
    """
    Save connector configuration to the current user's in-memory session.
    (Actual data pull is intentionally not implemented here.)
    """
    payload = {}
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {"value": payload}
    except Exception:
        payload = {}

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
        # best-effort only
        pass

    return JSONResponse({"status": "success", "connector": connector})

@router.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)):
    # Handle CSV upload
    # For demo, just return filename
    return JSONResponse({"filename": file.filename, "status": "uploaded"})

@router.get("/quickstart-csv")
def quickstart_csv():
    # Redirect to Quick Start with CSV section (frontend should handle this route)
    return RedirectResponse(url="/quickstart-csv-section")
