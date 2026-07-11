import io
from pathlib import Path

from PIL import Image
from starlette.datastructures import UploadFile
from torchvision import transforms

from app.ml.model_registry import ModelRegistry

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR.parent / "dis_models"

JEJAHITAN_CLASSES = [
    'Ceniga',
    'Ceper',
    'Ituk-ituk',
    'Kulit Peras',
    'Sampian Gantung',
    'Sampian Kwangen',
    'Sampian Padma',
    'Sampian Penyeneng',
    'Sampian Peras',
    'Sampian Plaus',
    'Sampian Sesayut',
    'Sampian Sri Keliki',
    'Taledan',
    'Tamas',
    'Lis Senjata',
    'Sampian Duras',
    'Sampian Penjor',
    'Sampian Soda'
]

JEJAHITAN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_image(image):
    if isinstance(image, UploadFile):
        image = Image.open(image.file).convert("RGB")
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Expected UploadFile, bytes, or PIL.Image.Image")
    return image


def _load_model(device=None):
    # kept for compatibility; ModelRegistry will call this when constructing the model
    import torch
    from app.ml.models.jejahitan_model import MobileViTv2WithCBAMOptimized

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileViTv2WithCBAMOptimized(num_classes=len(JEJAHITAN_CLASSES), pretrained=False)
    checkpoint = torch.load(MODEL_DIR / "model_jejahitan.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict_jejahitan(image, model_registry: ModelRegistry):
    import torch

    image = _load_image(image)
    print("========================")
    print(image.size)
    device = model_registry.device
    model = model_registry.get_model("jejahitan_model")
    print(type(model))
    print(model)

    input_tensor = JEJAHITAN_TRANSFORM(image).unsqueeze(0).to(device)

    print("=========================")
    print(input_tensor.shape)
    print(input_tensor.min().item())
    print(input_tensor.max().item())
    print(input_tensor.mean().item())
    print(input_tensor.std().item())

    with torch.no_grad():
        logits = model(input_tensor)
        print("====================")
        print(logits)
        print("====================")
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

        for idx, prob in zip(top5_idx[0], top5_prob[0]):
            print(
                idx.item(),
                JEJAHITAN_CLASSES[idx.item()],
                float(prob)
            )

    class_id = int(pred_idx.item())
    return {
        "class_id": class_id,
        "class_name": JEJAHITAN_CLASSES[class_id],
        "confidence": float(confidence.item()),
        "class_scores": probs.squeeze(0).tolist(),
    }
