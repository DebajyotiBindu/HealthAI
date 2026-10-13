
import streamlit as st

from src.prediction_service import predict_disease


st.set_page_config(
    page_title="Disease Prediction | HealthAI",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Disease Prediction")
st.write("Select a disease and enter the requested information.")

st.warning(
    "Academic prototype only. Predictions are not medical diagnoses. "
    "Feature-engineering assumptions must be validated against the "
    "original training code before results are relied upon."
)


def binary_field(label):
    return st.selectbox(label, ["No", "Yes"], key=label)


def binary_value(value):
    return 1 if value == "Yes" else 0


def show_result(disease, inputs):
    try:
        result = predict_disease(disease, inputs)
        prediction = int(result["prediction"])

        st.subheader("Prediction result")

        if prediction == 1:
            st.error(
                "The model predicted class 1. This is a positive "
                "model classification, not a confirmed diagnosis."
            )
        else:
            st.success(
                "The model predicted class 0. This does not rule out "
                "the disease or guarantee that a person is healthy."
            )

        probabilities = result["probability"]

        if probabilities is not None:
            st.write("Model output probabilities")
            st.json({
                label: f"{probability:.1%}"
                for label, probability in probabilities.items()
            })

        st.caption(
            "Probability values reflect the model's output and may "
            "not represent calibrated clinical risk."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


disease = st.selectbox(
    "Choose a disease",
    [
        "Alzheimer's",
        "Diabetes",
        "Liver Disease",
        "Lung Cancer",
        "Stroke",
    ],
)

st.divider()


# ---------------- ALZHEIMER'S ----------------
if disease == "Alzheimer's":
    st.subheader("Patient information")

    with st.form("alzheimers_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", 18, 110, 60)
            gender = st.selectbox("Gender (0/1)", [0, 1])
            ethnicity = st.number_input("Ethnicity code", 0, 10, 0)
            education = st.number_input("Education level code", 0, 10, 2)
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
            smoking = binary_field("Smoking")
            alcohol = st.number_input("Alcohol consumption", 0.0, 20.0, 0.0)
            activity = st.number_input("Physical activity", 0.0, 20.0, 5.0)
            diet = st.number_input("Diet quality", 0.0, 10.0, 5.0)
            sleep = st.number_input("Sleep quality", 0.0, 10.0, 5.0)
            family = binary_field("Family history of Alzheimer's")
            cardiovascular = binary_field("Cardiovascular disease")
            diabetes = binary_field("Diabetes")
            depression = binary_field("Depression")
            head_injury = binary_field("Head injury")
            hypertension = binary_field("Hypertension")

        with right:
            systolic = st.number_input("Systolic BP", 60, 260, 120)
            diastolic = st.number_input("Diastolic BP", 30, 160, 80)
            cholesterol = st.number_input("Total cholesterol", 50.0, 500.0, 200.0)
            ldl = st.number_input("LDL cholesterol", 0.0, 400.0, 100.0)
            hdl = st.number_input("HDL cholesterol", 0.0, 150.0, 50.0)
            triglycerides = st.number_input("Triglycerides", 0.0, 1500.0, 150.0)
            mmse = st.number_input("MMSE", 0.0, 30.0, 25.0)
            functional = st.number_input("Functional assessment", 0.0, 10.0, 5.0)
            memory = binary_field("Memory complaints")
            behavioral = binary_field("Behavioral problems")
            adl = st.number_input("ADL", 0.0, 10.0, 5.0)
            confusion = binary_field("Confusion")
            disorientation = binary_field("Disorientation")
            personality = binary_field("Personality changes")
            difficulty = binary_field("Difficulty completing tasks")
            forgetfulness = binary_field("Forgetfulness")

        submitted = st.form_submit_button(
            "Run Alzheimer's model",
            use_container_width=True,
        )

    if submitted:
        inputs = {
            "Age": age,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "EducationLevel": education,
            "BMI": bmi,
            "Smoking": binary_value(smoking),
            "AlcoholConsumption": alcohol,
            "PhysicalActivity": activity,
            "DietQuality": diet,
            "SleepQuality": sleep,
            "FamilyHistoryAlzheimers": binary_value(family),
            "CardiovascularDisease": binary_value(cardiovascular),
            "Diabetes": binary_value(diabetes),
            "Depression": binary_value(depression),
            "HeadInjury": binary_value(head_injury),
            "Hypertension": binary_value(hypertension),
            "SystolicBP": systolic,
            "DiastolicBP": diastolic,
            "CholesterolTotal": cholesterol,
            "CholesterolLDL": ldl,
            "CholesterolHDL": hdl,
            "CholesterolTriglycerides": triglycerides,
            "MMSE": mmse,
            "FunctionalAssessment": functional,
            "MemoryComplaints": binary_value(memory),
            "BehavioralProblems": binary_value(behavioral),
            "ADL": adl,
            "Confusion": binary_value(confusion),
            "Disorientation": binary_value(disorientation),
            "PersonalityChanges": binary_value(personality),
            "DifficultyCompletingTasks": binary_value(difficulty),
            "Forgetfulness": binary_value(forgetfulness),
        }
        show_result(disease, inputs)


# ---------------- DIABETES ----------------
elif disease == "Diabetes":
    with st.form("diabetes_form"):
        c1, c2 = st.columns(2)

        with c1:
            pregnancies = st.number_input("Pregnancies", 0, 30, 0)
            glucose = st.number_input("Glucose", 0.0, 400.0, 120.0)
            blood_pressure = st.number_input("Blood pressure", 0.0, 200.0, 80.0)
            skin = st.number_input("Skin thickness", 0.0, 100.0, 20.0)

        with c2:
            insulin = st.number_input("Insulin", 0.0, 1000.0, 80.0)
            bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
            pedigree = st.number_input(
                "Diabetes pedigree function", 0.0, 5.0, 0.5
            )
            age = st.number_input("Age", 18, 110, 30)

        submitted = st.form_submit_button(
            "Run diabetes model", use_container_width=True
        )

    if submitted:
        inputs = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age,
        }
        show_result(disease, inputs)


# ---------------- LIVER DISEASE ----------------
elif disease == "Liver Disease":
    with st.form("liver_form"):
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 110, 40)
            gender = st.selectbox("Gender (0/1)", [0, 1])
            bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
            alcohol = st.number_input(
                "Alcohol consumption", 0.0, 20.0, 0.0
            )
            smoking = st.selectbox("Smoking (0/1)", [0, 1])

        with c2:
            genetic = st.selectbox("Genetic risk (0/1)", [0, 1])
            activity = st.number_input(
                "Physical activity", 0.0, 20.0, 5.0
            )
            diabetes = st.selectbox("Diabetes (0/1)", [0, 1])
            hypertension = st.selectbox("Hypertension (0/1)", [0, 1])
            liver_test = st.number_input(
                "Liver function test value", 0.0, 1000.0, 30.0
            )

        submitted = st.form_submit_button(
            "Run liver disease model", use_container_width=True
        )

    if submitted:
        inputs = {
            "Age": age,
            "Gender": gender,
            "BMI": bmi,
            "AlcoholConsumption": alcohol,
            "Smoking": smoking,
            "GeneticRisk": genetic,
            "PhysicalActivity": activity,
            "Diabetes": diabetes,
            "Hypertension": hypertension,
            "LiverFunctionTest": liver_test,
        }
        show_result(disease, inputs)


# ---------------- LUNG CANCER ----------------
elif disease == "Lung Cancer":
    with st.form("lung_form"):
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.number_input("Age", 18, 110, 45)

        labels = [
            ("SMOKING", "Smoking"),
            ("YELLOW_FINGERS", "Yellow fingers"),
            ("ANXIETY", "Anxiety"),
            ("PEER_PRESSURE", "Peer pressure"),
            ("CHRONIC DISEASE", "Chronic disease"),
            ("FATIGUE", "Fatigue"),
            ("ALLERGY", "Allergy"),
            ("WHEEZING", "Wheezing"),
            ("ALCOHOL CONSUMING", "Alcohol consumption"),
            ("COUGHING", "Coughing"),
            ("SHORTNESS OF BREATH", "Shortness of breath"),
            ("SWALLOWING DIFFICULTY", "Swallowing difficulty"),
            ("CHEST PAIN", "Chest pain"),
        ]

        inputs = {"GENDER": gender, "AGE": age}

        for column, label in labels:
            inputs[column] = st.selectbox(
                label, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes",
                key=f"lung_{column}",
            )

        submitted = st.form_submit_button(
            "Run lung cancer model", use_container_width=True
        )

    if submitted:
        show_result(disease, inputs)


# ---------------- STROKE ----------------
elif disease == "Stroke":
    with st.form("stroke_form"):
        c1, c2 = st.columns(2)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            age = st.number_input("Age", 0, 110, 45)
            hypertension = st.selectbox("Hypertension", [0, 1])
            heart_disease = st.selectbox("Heart disease", [0, 1])
            married = st.selectbox("Ever married", ["No", "Yes"])
            work = st.selectbox(
                "Work type",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            )

        with c2:
            residence = st.selectbox("Residence type", ["Urban", "Rural"])
            glucose = st.number_input(
                "Average glucose level", 0.0, 500.0, 100.0
            )
            bmi = st.number_input("BMI", 0.0, 100.0, 25.0)
            smoking = st.selectbox(
                "Smoking status",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
            )

        submitted = st.form_submit_button(
            "Run stroke model", use_container_width=True
        )

    if submitted:
        inputs = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": married,
            "work_type": work,
            "Residence_type": residence,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking,
        }
        show_result(disease, inputs)
