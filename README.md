# HealthAI — AI-Powered Disease Prediction Platform

HealthAI is a machine-learning-based web application that integrates five disease prediction models into a unified Streamlit interface. Users can enter health-related information, generate model predictions, review previous predictions, and explore prediction analytics.

**This project was developed for academic and educational purposes.** Its outputs are not medical diagnoses and should not replace professional medical advice.

## Features

- **Multi-disease prediction:** Five disease-specific machine learning models.
- **Unified dashboard:** View integrated models, prediction statistics, and recent activity.
- **Model analytics:** Explore available model information and associated pipelines.
- **Prediction history:** Review previously recorded predictions.
- **Persistent storage:** Store prediction history locally using SQLite.
- **Modular architecture:** Separate model management, prediction processing, and history functionality.
- **Interactive interface:** Built with Streamlit.

## Supported Diseases

1. Alzheimer's Disease
2. Diabetes
3. Liver Disease
4. Lung Cancer
5. Stroke

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web application | Streamlit |
| Data processing | Pandas |
| Machine learning | Scikit-learn |
| Data storage | SQLite |
| Model serialization | Pickle (`.pkl`) |
| Data visualization | Streamlit charts |

## Project Structure

```text
final_year_project/
├── app.py
├── overview.py
├── pages/
│   ├── ... existing application pages
├── src/
│   ├── model_manager.py
│   ├── history_service.py
│   ├── prediction_service.py
│   └── ... other source files
├── models/
│   └── ... trained model artifacts
├── requirements.txt
└── README.md
```

*Note: The structure above is illustrative. Keep the actual filenames and directories from your project; do not rename existing files just to match this example.*

## Application Workflow

1. The user opens HealthAI and selects a disease prediction module.
2. The user provides the required input features.
3. The application processes the inputs and passes them to the corresponding trained model.
4. The model generates a predicted class.
5. The application displays the prediction and records the activity in the prediction history.
6. The dashboard and analytics pages present available prediction statistics and historical activity.

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies.
- pip
- Git (optional)

### 1. Clone or download the project

If the project is hosted in a Git repository:

```bash
git clone <your-repository-url>
cd final_year_project
```

Alternatively, open the project directory directly in your terminal.

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify model artifacts

Ensure all five trained model artifacts are available in the locations expected by the model manager. The model filenames and paths must match the configuration in `src/model_manager.py`.

### 5. Launch HealthAI

From the project root, run:

```bash
streamlit run app.py
```

Streamlit will provide a local URL, typically:

```text
http://localhost:8501
```

Open that URL in your browser.

## Usage

1. Launch the application.
2. Explore the dashboard and integrated model information.
3. Navigate to a disease-specific prediction page.
4. Enter the required input values and submit the form.
5. Review the generated prediction.
6. Visit Prediction History to review saved activity.
7. Open Model Analytics to inspect the available model information and analytics.

## Data Storage and Model Artifacts

Trained models are serialized as `.pkl` files and loaded by the model management module. Prediction history is maintained using SQLite.

Keep the trained model artifacts and their associated preprocessing pipelines consistent with the features used during training. Do not load pickle files from untrusted sources.

## Limitations

- Predictions depend on the quality, coverage, and limitations of the training datasets.
- Model outputs may be inaccurate and should not be interpreted as confirmed diagnoses.
- The application is an academic prototype, not a clinically validated medical device.
- Performance on new or unseen populations may differ from evaluation results.

## Future Improvements

- Expand model evaluation using independent test datasets.
- Improve input validation and error handling.
- Add more detailed model performance visualizations.
- Improve accessibility and user interface design.
- Evaluate model robustness, fairness, and generalization.

## Disclaimer

HealthAI is an educational machine learning project. It is not intended to diagnose, treat, cure, or prevent any disease. Always consult a qualified healthcare professional for medical concerns.

## Author

Developed as a final-year academic project.
