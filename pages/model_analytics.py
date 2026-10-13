
import streamlit as st
import pandas as pd

from src.model_manager import (
    MODEL_FILES,
    load_model,
    get_model_features,
    get_model_classes,
)

st.set_page_config(
    page_title="Model Analytics | HealthAI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Model Analytics")
st.caption(
    "Explore the five trained machine learning models integrated into HealthAI."
)

# --------------------------------------------------
# Load model information
# --------------------------------------------------

@st.cache_data
def get_model_summary():
    summaries = []

    for disease, filename in MODEL_FILES.items():
        model = load_model(disease)

        summaries.append({
            "Disease": disease,
            "Model File": filename,
            "Algorithm": type(model.named_steps["model"]).__name__,
            "Input Features": len(get_model_features(disease)),
            "Prediction Classes": ", ".join(
                str(c) for c in get_model_classes(disease)
            ),
        })

    return pd.DataFrame(summaries)


try:
    summary = get_model_summary()

    # --------------------------------------------------
    # Overview metrics
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Models", len(summary))

    with col2:
        st.metric(
            "Total Model Input Features",
            int(summary["Input Features"].sum()),
        )

    with col3:
        st.metric(
            "Different Algorithms",
            summary["Algorithm"].nunique(),
        )

    st.divider()

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------

    st.subheader("Model Overview")

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------
    # Feature explorer
    # --------------------------------------------------

    st.divider()
    st.subheader("Feature Explorer")

    selected_disease = st.selectbox(
        "Choose a disease model",
        list(MODEL_FILES.keys()),
    )

    selected_model = load_model(selected_disease)
    features = get_model_features(selected_disease)
    classes = get_model_classes(selected_disease)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Model Details")
        st.write("**Disease:**", selected_disease)
        st.write(
            "**Algorithm:**",
            type(selected_model.named_steps["model"]).__name__,
        )
        st.write("**Input feature count:**", len(features))
        st.write("**Prediction classes:**", classes)

        st.markdown("#### Pipeline Components")

        for step_name, step in selected_model.named_steps.items():
            st.write(f"• **{step_name}:** {type(step).__name__}")

    with col2:
        st.markdown("#### Expected Input Features")

        feature_df = pd.DataFrame({
            "No.": range(1, len(features) + 1),
            "Feature Name": features,
        })

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True,
            height=450,
        )

    # --------------------------------------------------
    # Important note on evaluation
    # --------------------------------------------------

    st.divider()
    st.subheader("Model Evaluation")

    st.info(
        "Accuracy, precision, recall, F1-score, and confusion matrices "
        "should be calculated using held-out test data or verified "
        "evaluation results from the original training process. "
        "They are not inferred from the saved model."
    )

except Exception as error:
    st.error(f"Unable to load model analytics: {error}")
    st.exception(error)
