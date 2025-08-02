import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

# ✅ Ensure data folder exists
os.makedirs("data/raw", exist_ok=True)
data_path = "data/raw/california.csv"

# ✅ Check if dataset exists, else download
if not os.path.exists(data_path):
    print("⚠️ Dataset not found. Downloading California Housing dataset...")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.to_csv(data_path, index=False)
    print(f"✅ Dataset downloaded and saved to {data_path}")
else:
    print(f"✅ Found dataset at {data_path}")

# ✅ Load data
data = pd.read_csv(data_path)
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# ✅ Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Start MLflow experiment
mlflow.set_experiment("California_Housing_Experiment")

# ✅ Models to train
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

best_model = None
best_rmse = float("inf")

for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        # Train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # ✅ Log parameters & metrics
        mlflow.log_param("model_type", model_name)
        if model_name == "DecisionTree":
            mlflow.log_param("max_depth", 5)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # ✅ Log model to MLflow
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} → RMSE: {rmse:.4f}, R2: {r2:.4f}")

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

# ✅ Save best model locally
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
print(f"✅ Best model saved as models/best_model.pkl with RMSE: {best_rmse:.4f}")
