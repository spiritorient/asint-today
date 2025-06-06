# main.py

import os
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse        # <-- import JSONResponse here
from pydantic import BaseModel, Field, validator
import httpx
from dotenv import load_dotenv

# ------------------------------------------------------------------
# 1. Load environment variables (including GUARDIAN_API_KEY)
# ------------------------------------------------------------------

load_dotenv()  # Reads .env in dev

GUARDIAN_API_KEY = os.getenv("GUARDIAN_API_KEY")
if not GUARDIAN_API_KEY:
    raise RuntimeError("Please set GUARDIAN_API_KEY in your environment or .env file.")

# ------------------------------------------------------------------
# 2. Define Pydantic models to validate incoming JSON
# ------------------------------------------------------------------

class SearchGuardianArgs(BaseModel):
    """
    Validate that OpenAI’s function call arguments match our schema exactly.
    """
    query: str = Field(..., description="Keyword(s) to search for (required).")
    page: int = Field(1, ge=1, description="Page number (default 1).")
    pageSize: int = Field(10, ge=1, le=50, description="Results per page (1–50).")
    fromDate: str | None = Field(None, description="YYYY-MM-DD or null.")
    toDate: str | None = Field(None, description="YYYY-MM-DD or null.")
    orderBy: str | None = Field(None, description="newest, oldest, or relevance.")

    @validator("fromDate", "toDate")
    def validate_date(cls, v):
        if v is None:
            return v
        parts = v.split("-")
        if len(parts) != 3 or not all(part.isdigit() for part in parts):
            raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class FunctionCall(BaseModel):
    """
    The structure OpenAI will POST when GPT does a function call.
    """
    name: str
    arguments: SearchGuardianArgs
    call_id: str


class FunctionCallOutput(BaseModel):
    """
    The structure we send back to OpenAI after fetching from The Guardian.
    """
    call_id: str
    output: dict


# ------------------------------------------------------------------
# 3. Create the FastAPI app
# ------------------------------------------------------------------

app = FastAPI(
    title="OpenAI → Guardian Search Webhook",
    version="1.0.0",
    description="Handles function calls for `search_guardian` by fetching real Guardian data."
)


@app.get("/", tags=["health"])
async def health_check():
    """
    Health check endpoint. Visit '/' in your browser or `curl` to ensure the service is running.
    """
    return {"status": "ok", "message": "search_guardian webhook is running"}


@app.post("/search_guardian", response_model=FunctionCallOutput, status_code=200)
async def search_guardian(call: FunctionCall, request: Request):
    """
    Main webhook endpoint:

    1. Validates incoming JSON (must match FunctionCall model).
    2. Builds the Guardian API request.
    3. Fetches data from content.guardianapis.com.
    4. Returns a JSON with {"call_id": "...", "output": { ... }}.
    """
    args = call.arguments

    # Build Guardian API URL and parameters
    base_url = "https://content.guardianapis.com/search"
    params = {
        "api-key": GUARDIAN_API_KEY,
        "q": args.query,
        "page": args.page,
        "page-size": args.pageSize,
        "format": "json"
    }
    if args.fromDate:
        params["from-date"] = args.fromDate
    if args.toDate:
        params["to-date"] = args.toDate
    if args.orderBy:
        params["order-by"] = args.orderBy

    # Fetch from The Guardian (async)
    async with httpx.AsyncClient(timeout=150) as client:
        resp = await client.get(base_url, params=params)

    if resp.status_code != 200:
        # If The Guardian returned an error, forward a 502
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Guardian API returned HTTP {resp.status_code}"
        )

    data = resp.json()
    if "response" not in data or "results" not in data["response"]:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Unexpected JSON structure from Guardian API"
        )

    # Extract only the fields we care about
    results_list = []
    for item in data["response"]["results"]:
        results_list.append({
            "id": item.get("id", ""),
            "webTitle": item.get("webTitle", ""),
            "webUrl": item.get("webUrl", ""),
            "webPublicationDate": item.get("webPublicationDate", "")
        })

    # Build the output payload
    output_payload = {
        "status": data["response"].get("status", ""),
        "total": data["response"].get("total", 0),
        "pageSize": data["response"].get("pageSize", 0),
        "currentPage": data["response"].get("currentPage", 0),
        "pages": data["response"].get("pages", 0),
        "orderBy": data["response"].get("orderBy", ""),
        "results": results_list
    }

    return FunctionCallOutput(call_id=call.call_id, output=output_payload)


# ------------------------------------------------------------------
# 4. Error handlers (fixed to use JSONResponse import)
# ------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error", "details": str(exc)}
    )