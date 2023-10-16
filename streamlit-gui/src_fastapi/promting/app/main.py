from fastapi import FastAPI
from app.api.promting import promting


app = FastAPI(
    openapi_url="/api/v1/promting/openapi.json",
    docs_url="/api/v1/promting/docs",
)

app.include_router(
    promting, prefix="/api/v1/promting", tags=["promting"]
)
