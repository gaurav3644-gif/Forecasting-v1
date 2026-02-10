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
def connect(connector: str, request: Request):
    # Here you would validate and attempt connection
    # For demo, just echo back the received data
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
