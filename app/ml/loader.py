from pathlib import Path

import tensorflow as tf
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR.parent / "dis_models"

wayang_model = tf.lite.Interpreter(
    model_path=str(MODEL_DIR / "model_wayang.tflite")
)
wayang_model.allocate_tensors()

ketupat_model = tf.lite.Interpreter(
    model_path=str(MODEL_DIR / "model_ketupat.tflite")
)
ketupat_model.allocate_tensors()

jejahitan_model = torch.load(
    MODEL_DIR / "model_jejahitan.pth",
    map_location="cpu"
)
jejahitan_model.eval()