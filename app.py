from fastapi import FastAPI
import uvicorn ##ASGI
from routers.routers import router

app = FastAPI()

app.include_router(router, tags=["asr"])