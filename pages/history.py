
import streamlit as st
import pandas as pd

from src.history_service import (
    get_prediction_history,
    clear_prediction_history,
)

st.set_page_config(
    page_title="Prediction History | HealthAI",
    page_icon="🕘",
    layout="wide",
)

st.title("🕘 Prediction History")
st.caption("Review previously generated predictions.")

history = get_prediction_history()

if not history:
    st.info(
        "No prediction history yet. Run a prediction first, "
        "and it will appear here."
    )
    st.stop()

# Filters
diseases = sorted({item["disease"] for item in history})

col1, col2 = st.columns([2, 1])

with col1:
    selected_disease = st.selectbox(
        "Filter by disease",
        ["All Diseases"] + diseases,
    )

with col2:
    st.metric("Total Saved Predictions", len(history))

filtered_history = history

if selected_disease != "All Diseases":
    filtered_history = [
        item for item in history
        if item["disease"] == selected_disease
    ]

# Summary table
table_data = []

for item in filtered_history:
    table_data.append({
        "ID": item["id"],
        "Date & Time": item["timestamp"],
        "Disease": item["disease"],
        "Predicted Class": item["prediction"],
    })

st.subheader("Saved Records")

st.dataframe(
    pd.DataFrame(table_data),
    use_container_width=True,
    hide_index=True,
)

# Record details
st.subheader("Prediction Details")

record_options = {
    f'#{item["id"]} — {item["disease"]} — {item["timestamp"]}':
    item
    for item in filtered_history
}

selected_record_label = st.selectbox(
    "Choose a record",
    list(record_options.keys()),
)

record = record_options[selected_record_label]

st.write("**Disease:**", record["disease"])
st.write("**Predicted class:**", record["prediction"])
st.write("**Timestamp:**", record["timestamp"])

if record["probabilities"] is not None:
    st.markdown("#### Model output probabilities")

    probability_data = [
        {
            "Class": label,
            "Probability (%)": round(float(value) * 100, 2),
        }
        for label, value in record["probabilities"].items()
    ]

    st.dataframe(
        pd.DataFrame(probability_data),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("View submitted input values"):
    st.json(record["input_data"])

st.divider()

st.warning(
    "Clearing history permanently deletes all saved prediction records "
    "from this local database."
)

if st.button("Clear All History", type="secondary"):
    st.session_state["confirm_clear_history"] = True

if st.session_state.get("confirm_clear_history", False):
    st.write("Are you sure you want to delete every saved record?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Yes, delete all records", type="primary"):
            clear_prediction_history()
            st.session_state["confirm_clear_history"] = False
            st.rerun()

    with col2:
        if st.button("Cancel"):
            st.session_state["confirm_clear_history"] = False
            st.rerun()
