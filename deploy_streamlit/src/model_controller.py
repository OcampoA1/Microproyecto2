from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class PredictionInputError(ValueError):
    """Error de validación para un texto que no puede clasificarse."""


@dataclass(frozen=True)
class Alternative:
    ods: int
    name: str
    probability: float


@dataclass(frozen=True)
class Prediction:
    ods: int
    name: str
    probability: float
    alternatives: tuple[Alternative, ...]
    low_confidence: bool


class ModelController:
    """Carga el artefacto y expone una interfaz estable de predicción."""

    def __init__(self, artifact: dict[str, Any], low_confidence_threshold: float = 0.40):
        required = {"schema_version", "model", "vectorizer", "ods_names", "metadata"}
        missing = required - set(artifact)
        if missing:
            raise ValueError(f"El artefacto no contiene las claves requeridas: {sorted(missing)}")
        if artifact["schema_version"] != 1:
            raise ValueError(f"Versión de artefacto no soportada: {artifact['schema_version']}")

        self.model = artifact["model"]
        self.vectorizer = artifact["vectorizer"]
        self.ods_names = {int(key): value for key, value in artifact["ods_names"].items()}
        self.metadata = artifact["metadata"]
        self.low_confidence_threshold = low_confidence_threshold

    @classmethod
    def from_file(cls, path: str | Path) -> "ModelController":
        model_path = Path(path)
        if not model_path.is_file():
            raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
        return cls(joblib.load(model_path))

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        cleaned = " ".join(str(text).strip().split())
        if not cleaned:
            raise PredictionInputError("Ingrese un texto antes de solicitar la clasificación.")
        if len(cleaned.split()) < 5:
            raise PredictionInputError("Escriba al menos cinco palabras con contenido.")
        if top_k < 1:
            raise ValueError("top_k debe ser mayor o igual que uno.")
        if self.vectorizer.transform([cleaned]).nnz == 0:
            raise PredictionInputError(
                "El texto no contiene términos reconocidos. Agregue información más específica."
            )

        probabilities = self.model.predict_proba([cleaned])[0]
        classes = self.model.classes_.astype(int)
        ranking = np.argsort(probabilities)[::-1][: min(top_k, len(classes))]
        alternatives = tuple(
            Alternative(
                ods=int(classes[index]),
                name=self.ods_names[int(classes[index])],
                probability=float(probabilities[index]),
            )
            for index in ranking
        )
        winner = alternatives[0]
        return Prediction(
            ods=winner.ods,
            name=winner.name,
            probability=winner.probability,
            alternatives=alternatives,
            low_confidence=winner.probability < self.low_confidence_threshold,
        )
