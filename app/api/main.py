from fastapi import FastAPI
from app.api.detection import router

app = FastAPI()

app.include_router(router)