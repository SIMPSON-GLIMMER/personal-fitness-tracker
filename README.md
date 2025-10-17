# personal-fitness-tracker


A Streamlit web application that predicts the number of calories burned based on user-provided fitness parameters such as age, BMI, heart rate, and exercise duration. The app uses a **Random Forest Regression** model trained on exercise and calorie datasets to make personalized predictions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://personal-fitness-tracking-system.streamlit.app/)

---

## 📊 Features

- Interactive sidebar for inputting personal data (Age, BMI, Heart Rate, etc.)
- Real-time prediction of calories burned (in kilocalories)
- Displays similar exercise records from the dataset
- Comparison stats to see where you stand among other users
- Clean, responsive UI built with Streamlit

---

## 🚀 Live Demo

Try it here:  
👉https://personal-fitness-tracking-system.streamlit.app/



## 🧠 Tech Stack

- **Python 3**
- **Streamlit**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn** (RandomForestRegressor)



## 🗂️ Project Structure



personal-fitness-tracker/

│

├── app.py                   # Main Streamlit app

├── calories.csv             # Calorie data

├── exercise.csv             # Exercise data

├── requirements.txt         # Project dependencies

└── README.md                # Project documentation


## ⚙️ Setup and Run Locally

1. **Clone the repository**
   
   git clone https://github.com/your-username/personal-fitness-tracker.git
   cd personal-fitness-tracker


2. **Install dependencies**
   
   pip install -r requirements.txt
   

3. **Run the app**

   streamlit run app.py
   

4. Open the link shown in the terminal (usually `http://localhost:8501`).


## 🧾 How It Works

1. The app merges `exercise.csv` and `calories.csv` datasets using `User_ID`.
2. BMI is calculated dynamically using weight and height.
3. The model (Random Forest Regressor) is trained on this processed dataset.
4. When you enter your details, the model predicts your expected calorie burn.


## 📈 Future Improvements

* Add user authentication and data history.
* Include more physiological parameters (e.g., steps, oxygen level).
* Deploy model backend separately via FastAPI for scalability.


Would you like me to also include a small `requirements.txt` file snippet you can paste into your repo (so Streamlit Cloud installs everything automatically)?
```
