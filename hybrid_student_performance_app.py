
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Generate mock dataset
def generate_mock_data(n=200):
    np.random.seed(42)
    data = pd.DataFrame({
        'Study Hours': np.random.randint(0, 15, n),
        'Sleep Hours': np.random.randint(3, 10, n),
        'Attendance Rate': np.random.randint(50, 100, n),
        'Class Participation': np.random.randint(1, 10, n),
        'Assignments Completed': np.random.randint(1, 10, n),
        'Gadget Usage (hrs)': np.random.randint(0, 12, n),
        'Likely to Pass': np.random.randint(0, 2, n)
    })
    return data

data = generate_mock_data()
X = data.drop("Likely to Pass", axis=1)
y = data["Likely to Pass"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Gradient Boosting Classifier
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)

# Deep Neural Network
model_dnn = Sequential([
    Dense(16, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_dnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
model_dnn.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
y_pred_dnn = (model_dnn.predict(X_test) > 0.5).astype("int32").flatten()

# Evaluation metrics
def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred)
    }

metrics_gb = get_metrics(y_test, y_pred_gb)
metrics_dnn = get_metrics(y_test, y_pred_dnn)

# Streamlit UI
st.title("Hybrid Student Performance Predictor")

st.sidebar.header("Enter Student Details")
study_hours = st.sidebar.slider("Study Hours per Day", 0, 15, 5)
sleep_hours = st.sidebar.slider("Sleep Hours per Day", 3, 10, 7)
attendance = st.sidebar.slider("Attendance Rate (%)", 50, 100, 85)
participation = st.sidebar.slider("Class Participation (1-10)", 1, 10, 5)
assignments = st.sidebar.slider("Assignments Completed (1-10)", 1, 10, 8)
gadget_use = st.sidebar.slider("Gadget Use (hrs/day)", 0, 12, 3)

model_choice = st.sidebar.selectbox("Select Model", ("Gradient Boosting", "Deep Neural Network"))

if st.button("Predict"):
    input_data = pd.DataFrame([[study_hours, sleep_hours, attendance, participation, assignments, gadget_use]],
                              columns=X.columns)
    input_scaled = scaler.transform(input_data)

    if model_choice == "Gradient Boosting":
        prediction = model_gb.predict(input_scaled)[0]
        metrics = metrics_gb
    else:
        prediction = (model_dnn.predict(input_scaled)[0] > 0.5).astype("int32")
        metrics = metrics_dnn

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("Likely to Pass")
    else:
        st.error("Needs Improvement")

    st.subheader(f"{model_choice} Model Evaluation Metrics")
    for metric, value in metrics.items():
        st.write(f"**{metric}:** {value:.2f}")
