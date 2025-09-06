import numpy as np
import pandas as pd
import warnings
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_att, ord_att, oneHot_cat):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    ord_pipeline = Pipeline([
        ("ordinal", OrdinalEncoder())
    ])
    oneHot_pipeline = Pipeline([
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_att),
        ("ord", ord_pipeline, ord_att),
        ("one", oneHot_pipeline, oneHot_cat)
    ])
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # ---------------- Training ----------------
    df = pd.read_csv("insurance.csv")
    
    # Train-test split BEFORE encoding
    X = df.drop(columns="charges")
    Y = df["charges"]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=42, shuffle=True, test_size=0.2
    )
    
    # Add bmi_category AFTER split
    for dataset in [X_train, X_test]:
        dataset["bmi_category"] = pd.cut(
            dataset["bmi"], bins=[0.0,18.5,25.0,29.9,np.inf],
            labels=["underweight","nomralweight","overweight","obesity"]
        )
    
    ord_cat = ["sex", "smoker"]
    one_cat = ["bmi_category", "region"]
    num_att = ["age", "bmi", "children"]
    
    pipeline = build_pipeline(num_att, ord_cat, one_cat)
    
    # Fit pipeline on training data
    X_train_prepared = pipeline.fit_transform(X_train)
    X_test_prepared = pipeline.transform(X_test)
    
    # Save X_test for inference
    X_test.to_csv("inputData.csv", index=False)
    
    # Train model
    model = RandomForestRegressor()
    model.fit(X_train_prepared, Y_train)
    
    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("✅ Congrats! Model is trained and test data saved.")
    
else:
    # ---------------- Inference ----------------
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    inputData = pd.read_csv("inputData.csv")
    
    # Transform test data with pipeline
    transformed_input = pipeline.transform(inputData)
    predictions = model.predict(transformed_input)
    
    inputData["predictedCharges"] = predictions
    inputData.to_csv("output.csv", index=False)
    
    print("✅ Inference done! Predictions saved to output.csv")
