import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Personal Fitness Tracker", layout="centered")

st.write("## Personal Fitness Tracker")
st.write("In this WebApp, you can estimate your predicted calories burned based on parameters like Age, Gender, BMI, etc.")

# Sidebar Inputs
st.sidebar.header("User Input Parameters:")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15.0, 40.0, 22.0)
    duration = st.sidebar.slider("Duration (minutes)", 0, 60, 20)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 36.0, 42.0, 38.0)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

# Display parameters
st.write("---")
st.header("Your Parameters:")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.005)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

# Add BMI column
exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
exercise_df["BMI"] = round(exercise_df["BMI"], 2)

# Train-test split
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=42)

# Prepare features and target
train_data = train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
test_data = test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]

train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

X_train = train_data.drop("Calories", axis=1)
y_train = train_data["Calories"]

# Model training
model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Align input with training columns
df = df.reindex(columns=X_train.columns, fill_value=0)

# Prediction
prediction = model.predict(df)[0]

st.write("---")
st.header("Predicted Calories Burned:")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.005)
st.success(f"{round(prediction, 2)} kilocalories")

# Similar results
st.write("---")
st.header("Similar Results:")
lower, upper = prediction - 10, prediction + 10
similar = exercise_df[(exercise_df["Calories"] >= lower) & (exercise_df["Calories"] <= upper)]

if len(similar) > 0:
    st.write(similar.sample(min(5, len(similar))))
else:
    st.write("No similar records found.")

# General comparisons
st.write("---")
st.header("General Information:")

age_pct = (exercise_df["Age"] < df["Age"].values[0]).mean() * 100
dur_pct = (exercise_df["Duration"] < df["Duration"].values[0]).mean() * 100
hr_pct = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).mean() * 100
temp_pct = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).mean() * 100

st.write(f"You are older than {age_pct:.2f}% of other people.")
st.write(f"Your exercise duration is higher than {dur_pct:.2f}% of other people.")
st.write(f"Your heart rate is higher than {hr_pct:.2f}% of other people during exercise.")
st.write(f"Your body temperature is higher than {temp_pct:.2f}% of other people during exercise.")

