
import json
import re
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.models.full_model import CrossModalRetrievalModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "real_fold1" / "best_model.pth"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flickr30k"
    / "vocabularies"
    / "fold_1_vocab.json"
)

MAX_LENGTH = 34


class ModelService:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.vocabulary = None
        self.checkpoint_info = {}

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def load(self):
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {CHECKPOINT_PATH}"
            )

        if not VOCAB_PATH.is_file():
            raise FileNotFoundError(
                f"Vocabulary not found: {VOCAB_PATH}"
            )

        with VOCAB_PATH.open("r", encoding="utf-8") as file:
            self.vocabulary = json.load(file)

        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=self.device,
            weights_only=True,
        )

        vocab_size = checkpoint["vocab_size"]

        if vocab_size != len(self.vocabulary):
            raise ValueError(
                "Checkpoint vocabulary size does not match "
                "the vocabulary file."
            )

        if checkpoint["model_type"] != "ResNet50-LSTM":
            raise ValueError(
                f"Unexpected model type: {checkpoint['model_type']}"
            )

        self.model = CrossModalRetrievalModel(
            vocab_size=vocab_size,
            encoder_type="lstm",
            embedding_dim=300,
            hidden_dim=512,
            num_layers=1,
            dropout=0.2,
            bidirectional=False,
            use_attention=True,
            pretrained=False,
            unfreeze_last_block=False,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint_info = {
            "epoch": checkpoint["epoch"],
            "fold": checkpoint["fold"],
            "model_type": checkpoint["model_type"],
            "train_loss": checkpoint["train_loss"],
            "val_loss": checkpoint["val_loss"],
        }

        return self

    def encode_caption(self, caption):
        if not self.is_loaded:
            raise RuntimeError("Model service is not loaded.")

        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Caption must be a non-empty string.")

        tokens = re.findall(r"\w+|[^\w\s]", caption.lower())

        encoded = [
            self.vocabulary["<BOS>"],
            *[
                self.vocabulary.get(
                    token,
                    self.vocabulary["<UNK>"],
                )
                for token in tokens
            ],
            self.vocabulary["<EOS>"],
        ]

        if len(encoded) > MAX_LENGTH:
            encoded = encoded[:MAX_LENGTH]
            encoded[-1] = self.vocabulary["<EOS>"]

        padding_id = self.vocabulary["<PAD>"]

        encoded += [padding_id] * (MAX_LENGTH - len(encoded))

        return torch.tensor(
            [encoded],
            dtype=torch.long,
            device=self.device,
        )

    @torch.inference_mode()
    def get_text_embedding(self, caption):
        if not self.is_loaded:
            raise RuntimeError("Model service is not loaded.")

        tokens = self.encode_caption(caption)
        embedding = self.model.encode_text(tokens)

        return embedding

    def preprocess_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL Image.")

        image = image.convert("RGB")

        tensor = self.image_transform(image).unsqueeze(0)

        return tensor.to(self.device)

    @torch.inference_mode()
    def get_image_embedding(self, image):
        if not self.is_loaded:
            raise RuntimeError("Model service is not loaded.")

        image_tensor = self.preprocess_image(image)
        embedding = self.model.encode_image(image_tensor)

        return embedding

    @property
    def is_loaded(self):
        return self.model is not None and self.vocabulary is not None
