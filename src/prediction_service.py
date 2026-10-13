
from unittest import result

import pandas as pd
from src.history_service import save_prediction

from src.model_manager import load_model


# Provisional feature-engineering assumptions.
# Verify these against the original model-training notebook.
DIABETES_GLUCOSE_THRESHOLD = 140
STROKE_GLUCOSE_THRESHOLD = 140
STROKE_HIGH_BMI_THRESHOLD = 25
ALZHEIMERS_LOW_MMSE_THRESHOLD = 24


def age_group(age):
    if age < 30:
        return "Young"
    if age < 45:
        return "Middle-aged"
    if age < 60:
        return "Senior"
    return "Elderly"


def bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def _binary_risk_count(values):
    """Count positive risk flags."""
    return sum(int(value) for value in values)


def build_features(disease, inputs):
    """Build one input row with the exact expected feature names."""

    if disease == "Alzheimer's":
        row = dict(inputs)
        age = float(row["Age"])
        mmse = float(row["MMSE"])

        row["Age_Squared"] = age ** 2
        row["Age_Group"] = age_group(age)
        row["Low_MMSE_Flag"] = int(
            mmse < ALZHEIMERS_LOW_MMSE_THRESHOLD
        )
        row["Age_MMSE_Interaction"] = age * mmse

    elif disease == "Diabetes":
        row = dict(inputs)
        age = float(row["Age"])
        glucose = float(row["Glucose"])
        bmi = float(row["BMI"])

        row["BMI_Category"] = bmi_category(bmi)
        row["Age_Group"] = age_group(age)
        row["High_Glucose_Flag"] = int(
            glucose >= DIABETES_GLUCOSE_THRESHOLD
        )
        row["BMI_Glucose_Interaction"] = bmi * glucose
        row["Age_Glucose_Interaction"] = age * glucose
        row["Age_BMI_Interaction"] = age * bmi
        row["Risk_Factor_Count"] = _binary_risk_count([
            glucose >= DIABETES_GLUCOSE_THRESHOLD,
            bmi >= 25,
            age >= 45,
        ])

    elif disease == "Liver Disease":
        row = dict(inputs)

    elif disease == "Lung Cancer":
        row = dict(inputs)
        risk_columns = [
            "SMOKING",
            "YELLOW_FINGERS",
            "ANXIETY",
            "PEER_PRESSURE",
            "CHRONIC DISEASE",
            "FATIGUE",
            "ALLERGY",
            "WHEEZING",
            "ALCOHOL CONSUMING",
            "COUGHING",
            "SHORTNESS OF BREATH",
            "SWALLOWING DIFFICULTY",
            "CHEST PAIN",
        ]
        row["RISK_FACTOR_COUNT"] = _binary_risk_count(
            [row[column] for column in risk_columns]
        )

    elif disease == "Stroke":
        row = dict(inputs)
        age = float(row["age"])
        glucose = float(row["avg_glucose_level"])
        bmi = float(row["bmi"])
        hypertension = int(row["hypertension"])
        heart_disease = int(row["heart_disease"])

        row["age_squared"] = age ** 2
        row["age_group"] = age_group(age)
        row["high_glucose_flag"] = int(
            glucose >= STROKE_GLUCOSE_THRESHOLD
        )
        row["high_bmi_flag"] = int(
            bmi >= STROKE_HIGH_BMI_THRESHOLD
        )
        row["cardiovascular_risk_count"] = (
            hypertension + heart_disease
        )
        row["age_hypertension_interaction"] = age * hypertension
        row["age_heart_disease_interaction"] = age * heart_disease
        row["glucose_bmi_interaction"] = glucose * bmi

    else:
        raise ValueError(f"Unsupported disease: {disease}")

    return row


def predict_disease(disease, inputs):
    """Return the model's class prediction and probability, if available."""

    model = load_model(disease)
    features = build_features(disease, inputs)

    # Match the original model's expected feature order.
    expected_columns = list(model.feature_names_in_)
    missing = [col for col in expected_columns if col not in features]

    if missing:
        raise ValueError(
            f"Missing required features for {disease}: {missing}"
        )

    frame = pd.DataFrame(
        [[features[col] for col in expected_columns]],
        columns=expected_columns,
    )

    prediction = model.predict(frame)[0]

    result = {
        "prediction": prediction,
        "probability": None,
        "classes": model.classes_.tolist(),
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        result["probability"] = {
            str(label): float(probability)
            for label, probability in zip(
                model.classes_, probabilities
            )
        }

    save_prediction(disease, inputs, result)
    return result
