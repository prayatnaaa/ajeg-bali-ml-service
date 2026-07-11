from pathlib import Path
import threading

MODEL_DIR = Path(__file__).resolve().parents[1].parent / "dis_models"
MODEL_PATH_KETUPAT = MODEL_DIR / "model_ketupat.tflite"
MODEL_PATH_WAYANG = MODEL_DIR / "model_wayang.tflite"
MODEL_PATH_JEJAHITAN = MODEL_DIR / "model_jejahitan.pth"


class ModelRegistry:
    def __init__(self):
        self._device = None
        self._models = {}
        self._locks = {}
        self._lock = threading.Lock()

    @property
    def device(self):
        return self._device

    def load_models(self):
        with self._lock:
            if self._models:
                return

            self._device = self._detect_device()
            self._models["ketupat_interpreter"] = self._create_ketupat_interpreter()
            self._models["wayang_interpreter"] = self._create_wayang_interpreter()
            self._models["jejahitan_model"] = self._create_jejahitan_model()

            self._locks["ketupat_interpreter"] = threading.Lock()
            self._locks["wayang_interpreter"] = threading.Lock()
            self._locks["jejahitan_model"] = threading.Lock()

    def cleanup(self):
        self._models.clear()
        self._locks.clear()

    def get_model(self, name):
        return self._models.get(name)

    def get_lock(self, name):
        return self._locks.get(name)

    def _detect_device(self):
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_ketupat_interpreter(self):
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH_KETUPAT))
        interpreter.allocate_tensors()
        return interpreter

    def _create_wayang_interpreter(self):
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH_WAYANG))
        interpreter.allocate_tensors()
        return interpreter

    def _create_jejahitan_model(self):
        import torch
        from app.ml.jejahitan_classification import _load_model

        return _load_model(device=self._device)
