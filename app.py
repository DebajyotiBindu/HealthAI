
import streamlit as st
import pandas as pd

from src.history_service import get_prediction_history
from src.model_manager import MODEL_FILES, load_model

st.set_page_config(
    page_title="HealthAI | Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 HealthAI")
st.markdown("### Unified Machine Learning Disease Prediction Platform")
st.write(
    "Explore the integrated models, generate predictions, "
    "and review your local prediction history."
)

st.divider()

# --------------------------------------------------
# Load dashboard data
# --------------------------------------------------

try:
    history = get_prediction_history()

    total_models = len(MODEL_FILES)
    total_predictions = len(history)

    unique_diseases = len({
        item["disease"] for item in history
    })

    latest_prediction = (
        history[0]["timestamp"] if history else "No predictions yet"
    )

    # --------------------------------------------------
    # Summary cards
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Integrated Models", total_models)

    with col2:
        st.metric("Total Predictions", total_predictions)

    with col3:
        st.metric("Diseases Explored", unique_diseases)

    with col4:
        st.metric(
            "Available Model Pipelines",
            sum(1 for disease in MODEL_FILES if load_model(disease)),
        )

    st.divider()

    # --------------------------------------------------
    # Disease-wise activity
    # --------------------------------------------------

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("📈 Prediction Activity")

        if history:
            counts = (
                pd.DataFrame(history)
                .groupby("disease")
                .size()
                .reset_index(name="Predictions")
                .sort_values("Predictions", ascending=False)
            )

            st.bar_chart(
                counts,
                x="disease",
                y="Predictions",
                use_container_width=True,
            )
        else:
            st.info(
                "Your disease-wise activity chart will appear "
                "after you make your first prediction."
            )

    with right:
        st.subheader("🕘 Latest Activity")
        st.caption(f"Most recent saved prediction: {latest_prediction}")

        if history:
            recent = []

            for item in history[:5]:
                recent.append({
                    "Disease": item["disease"],
                    "Predicted Class": item["prediction"],
                    "Timestamp": item["timestamp"],
                })

            st.dataframe(
                pd.DataFrame(recent),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No saved predictions yet.")

    st.divider()

    # --------------------------------------------------
    # Model directory
    # --------------------------------------------------

    st.subheader("🧠 Integrated Disease Models")

    columns = st.columns(3)

    for index, (disease, filename) in enumerate(MODEL_FILES.items()):
        with columns[index % 3]:
            with st.container(border=True):
                st.markdown(f"**{disease}**")
                st.caption(filename)

                try:
                    model = load_model(disease)
                    estimator = model.named_steps.get("model")
                    algorithm = (
                        type(estimator).__name__
                        if estimator is not None
                        else "Pipeline"
                    )
                    st.write(f"Algorithm: {algorithm}")
                    st.success("Model loaded")
                except Exception:
                    st.error("Model could not be loaded")

    st.divider()

    # --------------------------------------------------
    # Navigation guide
    # --------------------------------------------------

    st.subheader("🚀 Explore HealthAI")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Make a Prediction**")
        st.write("Enter model inputs and view its output.")
        st.caption("Open Predict from the sidebar.")

    with col2:
        st.markdown("**Model Analytics**")
        st.write("Inspect algorithms, input features, and pipelines.")
        st.caption("Open Model Analytics from the sidebar.")

    with col3:
        st.markdown("**Prediction History**")
        st.write("Review saved predictions and their input values.")
        st.caption("Open Prediction History from the sidebar.")

    st.divider()

    st.caption(
        "Academic prototype only. Outputs are not medical diagnoses "
        "and should not guide treatment decisions."
    )

except Exception as error:
    st.error("The dashboard could not load its data.")
    st.exception(error)

