from fastapi import FastAPI

from app.api.routes import router
from app.core.storage import ensure_bucket_exists
from app.middleware import SimpleRateLimitMiddleware

app = FastAPI(title="HumanX Data Factory API")
app.add_middleware(SimpleRateLimitMiddleware)


@app.on_event("startup")
def startup_event():
    ensure_bucket_exists()


app.include_router(router)
