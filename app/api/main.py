import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.prediction import router
from app.ml.model_registry import ModelRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.model_registry = ModelRegistry()
	app.state.model_registry.load_models()
	try:
		yield
	finally:
		app.state.model_registry.cleanup()

app = FastAPI(lifespan=lifespan)

app.include_router(router)