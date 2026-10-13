
from pathlib import Path
import pickle

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "MODELS"

MODEL_FILES = {
    "Alzheimer's": "alzheimers_best_model.pkl",
    "Diabetes": "diabetes_best_model.pkl",
    "Liver Disease": "liver_disease_best_model.pkl",
    "Lung Cancer": "lung_cancer_best_model.pkl",
    "Stroke": "stroke_best_model.pkl",
}


@st.cache_resource
def load_model(disease_name: str):
    """Load and cache one trusted, trained model."""

    if disease_name not in MODEL_FILES:
        raise ValueError(f"Unknown disease: {disease_name}")

    model_path = MODEL_DIR / MODEL_FILES[disease_name]

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    with model_path.open("rb") as file:
        model = pickle.load(file)

    return model


def get_model_features(disease_name: str) -> list[str]:
    """Return the input feature names expected by the model."""

    model = load_model(disease_name)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            f"Feature names are unavailable for {disease_name}."
        )

    return list(model.feature_names_in_)


def get_model_classes(disease_name: str) -> list:
    """Return the classes the model can predict."""

    model = load_model(disease_name)

    if not hasattr(model, "classes_"):
        raise ValueError(
            f"Output classes are unavailable for {disease_name}."
        )

    return model.classes_.tolist()
