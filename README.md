# Insurance Charges Prediction Web App

A Machine Learning project to predict insurance charges based on personal attributes like age, BMI, gender, smoking status, number of children, and region. The project uses a **Random Forest Regressor** model with preprocessing pipelines, and exposes a **web interface** built with **Flask, HTML, CSS, and JS** for interactive predictions.

---

## **Project Structure**

insurance-ml-app/
│
├── app.py # Flask backend
├── model.pkl # Trained Random Forest model
├── pipeline.pkl # Preprocessing pipeline
├── insurance.csv # Original dataset
├── inputData.csv # Test split data
├── templates/
│ └── index.html # HTML frontend
├── static/
│ ├── style.css # CSS styling
│ └── script.js # Optional JS
├── README.md # This file


---

## **Features Used**

- **Numerical:** `age`, `bmi`, `children`
- **Ordinal/Categorical:** `sex`, `smoker`
- **One-Hot Encoded Categorical:** `region`, `bmi_category`  

`bmi_category` is derived from `bmi`:

| Category        | BMI Range      |
|-----------------|----------------|
| underweight     | 0 – 18.5       |
| normalweight    | 18.5 – 25      |
| overweight      | 25 – 29.9      |
| obesity         | 30+            |

---

## **Technologies Used**

- Python 3.x  
- Pandas, NumPy  
- Scikit-learn (Pipeline, ColumnTransformer, RandomForestRegressor)  
- Joblib (model and pipeline serialization)  
- Flask (Web interface)  
- HTML, CSS, JS (Frontend)

---

## **Setup Instructions**

1. **Clone the repository**
```bash
git clone https://github.com/<your-username>/insurance-ml-app.git
cd insurance-ml-app
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
pip install -r requirements.txt
Flask
numpy
pandas
scikit-learn
joblib
Flask
numpy
pandas
scikit-learn
joblib
How it Works

The dataset is split into training and test sets first (train_test_split).

Preprocessing is applied using a pipeline:

Numerical features: SimpleImputer + StandardScaler

Ordinal features: OrdinalEncoder

Categorical features: OneHotEncoder with handle_unknown="ignore"

The Random Forest Regressor is trained on the training set.

Test data is saved to inputData.csv for inference.

Flask UI allows users to input new data, preprocesses it using the saved pipeline, and predicts insurance charges.

How to Use

Run the app using Flask.

Enter input values in the web form:

Age, BMI, Children

Sex (Male/Female)

Smoker (Yes/No)

Region (Northeast, Northwest, Southeast, Southwest)

Click Predict.

The predicted insurance charges will be displayed.

Files Explanation

app.py – Flask backend, loads model & pipeline, handles inference.

model.pkl – Trained Random Forest model saved using Joblib.

pipeline.pkl – Preprocessing pipeline for transforming input data.

inputData.csv – Test split from original dataset (used for testing or demo).

templates/index.html – Web form for input.

static/style.css – Styles the frontend.

static/script.js – Optional JS for extra functionality.

Future Improvements

Add multi-regressor comparison for better predictions.

Deploy the app on Heroku / AWS / Streamlit Cloud.

Add visualizations like feature importance or BMI distribution.

Store predictions in a database for analytics.

License

This project is open-source. Feel free to use and modify.


---

I can also **create a ready-to-use `requirements.txt`** and **update the README with commands to retrain the model** automatically if `model.pkl` is missing.  

Do you want me to do that next?
