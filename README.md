# ❤️ Heart Disease Risk Predictor

An AI-powered **Heart Disease Risk Prediction** web application built using **Python, Streamlit, Pandas, Scikit-learn, Joblib, and Plotly**.

The application allows users to enter clinical and demographic information and uses a trained **Logistic Regression** machine learning model to estimate the risk of heart disease.

> ⚠️ **Disclaimer:** This application is intended for educational and informational purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.

---

## 📌 Features

* ❤️ Heart disease risk prediction
* 🤖 Machine learning prediction using Logistic Regression
* 📊 Interactive risk probability gauge
* 📋 Risk factor analysis
* 🧑‍⚕️ Clinical and demographic input fields
* 📈 Interactive Plotly visualization
* 🎨 Custom Streamlit interface
* 🧭 Sidebar navigation
* ⚡ Real-time predictions
* 🔐 Model and preprocessing files loaded using Joblib

---

## 🛠️ Technologies Used

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Programming language         |
| Streamlit             | Web application interface    |
| Pandas                | Data manipulation            |
| NumPy                 | Numerical operations         |
| Scikit-learn          | Machine learning             |
| Joblib                | Saving and loading ML models |
| Plotly                | Interactive visualizations   |
| streamlit-option-menu | Sidebar navigation           |

---

## 🧠 Machine Learning Model

The application uses a **Logistic Regression** classification model.

The model predicts whether the provided patient information indicates:

* `0` → Low Risk
* `1` → High Risk

The model uses the following features:

### Demographic Features

* Age
* Sex

### Clinical Measurements

* Resting Blood Pressure
* Cholesterol
* Fasting Blood Sugar
* Maximum Heart Rate
* Oldpeak

### ECG and Symptoms

* Chest Pain Type
* Resting ECG
* Exercise-Induced Angina
* ST Slope

A total of **11 input features** are used by the application.

---

## 📂 Project Structure

```text
HeartDisease/
│
├── Prediction Application/
│   │
│   ├── app.py
│   ├── LogisticRegression.pkl
│   ├── scaler.pkl
│   ├── columns.pkl
│   ├── requirements.txt
│   └── README.md
│
└── ...
```

### Important Files

**`app.py`**

Contains the Streamlit application and prediction logic.

**`LogisticRegression.pkl`**

Contains the trained Logistic Regression model.

**`scaler.pkl`**

Contains the fitted scaler used to transform input data before prediction.

**`columns.pkl`**

Contains the feature names and their required order.

**`requirements.txt`**

Contains the Python packages required to run the application.

---

## ⚙️ How the Application Works

```text
User Input
    ↓
Data Preparation
    ↓
One-Hot Encoded Features
    ↓
Feature Column Ordering
    ↓
Feature Scaling
    ↓
Logistic Regression Model
    ↓
Prediction + Probability
    ↓
Risk Result
```

The application first collects information from the user.

The input is then converted into a Pandas DataFrame and arranged according to the feature columns used during model training.

The saved scaler transforms the input into the same scale used during training.

Finally, the Logistic Regression model predicts the heart disease risk and probability.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
```

Move into the project directory:

```bash
cd heart-disease-prediction
```

### 2. Install Required Libraries

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` file, you can install the main dependencies manually:

```bash
pip install streamlit pandas numpy joblib plotly streamlit-option-menu scikit-learn
```

---

## ▶️ Run the Application

Run the following command from the project directory:

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 🖥️ Application Sections

### 🔮 Prediction

Users enter their:

* Age
* Sex
* Blood pressure
* Cholesterol
* Blood sugar
* Maximum heart rate
* Oldpeak
* Chest pain type
* Resting ECG
* Exercise angina
* ST slope

After clicking **Predict Heart Disease Risk**, the application displays the prediction and risk probability.

### 📋 Risk Factors

Provides information about:

* Modifiable risk factors
* Non-modifiable risk factors
* Lifestyle recommendations
* Important health metrics

### ℹ️ About

Provides information about:

* The machine learning model
* Features used
* How the application works
* Technical details
* Important medical disclaimer

---

## 📊 Prediction Output

The application provides:

### Low Risk

If the model predicts:

```text
0
```

the application displays a **Low Risk of Heart Disease** message.

### High Risk

If the model predicts:

```text
1
```

the application displays a **High Risk of Heart Disease** message and identifies possible risk factors based on the entered information.

The application also displays a **risk probability gauge** using Plotly.

---

## 📦 Model Files

The trained model and preprocessing objects are saved using Joblib:

```python
joblib.dump(model, "LogisticRegression.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(columns, "columns.pkl")
```

They are loaded in the Streamlit application using:

```python
model = joblib.load("LogisticRegression.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
```

---

## 📋 Requirements

Example `requirements.txt`:

```text
streamlit
pandas
numpy
joblib
plotly
streamlit-option-menu
scikit-learn
```

---

## 🔒 Disclaimer

This application is developed for **educational and informational purposes**.

The prediction generated by this application should **not be considered a medical diagnosis**. Machine learning predictions may contain errors and should not be used as a substitute for professional medical evaluation.

If you have concerns about your health or heart disease risk, consult a qualified healthcare professional.

---

## 👨‍💻 Project Purpose

This project demonstrates how a machine learning classification model can be integrated into an interactive web application.

It covers:

* Data preprocessing
* Categorical encoding
* Feature scaling
* Machine learning model training
* Model evaluation
* Model serialization with Joblib
* Streamlit application development
* Interactive data visualization

---

## ❤️ Future Improvements

Possible future improvements include:

* Compare multiple machine learning models
* Improve model performance
* Add more health-related features
* Add model explainability
* Add prediction history
* Deploy the application online
* Add authentication and user accounts
* Improve accessibility and responsive design

---

## ⭐ Acknowledgement

This project was developed as a learning project to explore **Machine Learning and Streamlit application development**.

---

## 📜 License

This project is available for educational purposes.
