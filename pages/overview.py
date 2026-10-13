
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="HealthAI | Overview",
    page_icon="🩺",
    layout="wide",
)

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = PROJECT_DIR / "pages"

st.title("🩺 HealthAI")
st.subheader("AI-Powered Disease Prediction Platform")

st.markdown(
    """
    Welcome to HealthAI! Explore disease prediction models,
    review prediction history, and inspect model analytics
    from one place.
    """
)

st.divider()

# Main dashboard
st.subheader("🏠 Main Dashboard")

st.write(
    "View prediction statistics, recent activity, and the "
    "status of integrated disease models."
)

st.page_link(
    "app.py",
    label="Open Main Dashboard",
    icon="🏠",
    use_container_width=True,
)

st.divider()

# Discover existing application pages
st.subheader("🚀 Explore HealthAI Pages")

page_files = sorted(
    [
        file
        for file in PAGES_DIR.glob("*.py")
        if file.resolve() != Path(__file__).resolve()
        and not file.name.startswith("_")
    ],
    key=lambda file: file.stem.lower(),
)

if page_files:
    columns = st.columns(2)

    for index, page_file in enumerate(page_files):
        page_name = page_file.stem.replace("_", " ").replace("-", " ").title()

        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"### {page_name}")
                st.write(f"Open the {page_name} section of HealthAI.")

                st.page_link(
                    str(Path("pages") / page_file.name),
                    label=f"Open {page_name}",
                    icon="➡️",
                    use_container_width=True,
                )
else:
    st.info(
        "No other page files were found in the pages directory. "
        "Check that your existing pages are saved there."
    )

st.divider()

st.subheader("🧠 Supported Disease Models")

disease_cols = st.columns(3)

diseases = [
    "Alzheimer's Disease",
    "Diabetes",
    "Liver Disease",
    "Lung Cancer",
    "Stroke",
]

for index, disease in enumerate(diseases):
    with disease_cols[index % 3]:
        st.markdown(f"- {disease}")

st.divider()

st.warning(
    "HealthAI is an academic machine learning project. "
    "Predictions are educational estimates, not medical diagnoses. "
    "Consult a qualified healthcare professional for medical advice."
)
