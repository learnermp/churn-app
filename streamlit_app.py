import streamlit as st
import pickle

st.title("Customer Churn Prediction")
st.write("This project demonstrates how a trained ML model can be exposed through a simple UI using Streamlit for real-time prediction.")
# Load model ONLY once
@st.cache_resource
def load_model():
    with open('model/model-churn.bin', 'rb') as f:
        return pickle.load(f)

dv, model = load_model()

# Inputs
gender = st.selectbox("Gender", ["male", "female"])
seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["yes", "no"])
dependents = st.selectbox("Dependents", ["yes", "no"])
tenure = st.slider("Tenure", 0, 72)
phoneservice = st.selectbox("Phone Service", ["yes", "no"])
multiplelines = st.selectbox("Multiple Lines", ["yes", "no"])
internetservice = st.selectbox("Internet Service", ["dsl", "fiber_optic", "no"])
onlinesecurity = st.selectbox("Online Security", ["yes", "no"])
onlinebackup = st.selectbox("Online Backup", ["yes", "no"])
deviceprotection = st.selectbox("Device Protection", ["yes", "no"])
techsupport = st.selectbox("Tech Support", ["yes", "no"])
streamingtv = st.selectbox("Streaming TV", ["yes", "no"])
streamingmovies = st.selectbox("Streaming Movies", ["yes", "no"])
contract = st.selectbox("Contract", ["month-to-month", "one_year", "two_year"])
paperlessbilling = st.selectbox("Paperless Billing", ["yes", "no"])
paymentmethod = st.selectbox(
    "Payment Method",
    ["electronic_check", "mailed_check", "bank_transfer_(automatic)", "credit_card_(automatic)"]
)
monthlycharges = st.number_input("Monthly Charges", value=50.0)
totalcharges = st.number_input("Total Charges", value=1000.0)

if st.button("Predict"):
    customer = {
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": multiplelines,
        "internetservice": internetservice,
        "onlinesecurity": onlinesecurity,
        "onlinebackup": onlinebackup,
        "deviceprotection": deviceprotection,
        "techsupport": techsupport,
        "streamingtv": streamingtv,
        "streamingmovies": streamingmovies,
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges
    }

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    churn = y_pred >= 0.5

    st.subheader("Result")
    st.write(f"Churn Probability: {y_pred:.3f}")

    if churn:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is likely to stay ")