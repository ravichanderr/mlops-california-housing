from datetime import datetime
import logging
import sqlite3

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator

# ✅ Set up logging to a file
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ✅ Set up SQLite database
conn = sqlite3.connect("logs/predictions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        input_data TEXT,
        prediction REAL,
        timestamp TEXT
    )
    """
)
conn.commit()

# ✅ Load model and feature names
model = joblib.load("models/best_model.pkl")
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
    'Population', 'AveOccup', 'Latitude', 'Longitude'
]


class HousingData(BaseModel):
    features: list


app = FastAPI(title="California Housing API with Logging")


@app.get("/")
def home():
    return {"message": "California Housing API is running! Use /predict to make predictions."}


@app.post("/predict")
def predict(data: HousingData):
    df_input = pd.DataFrame([data.features], columns=feature_names)
    prediction = model.predict(df_input).tolist()[0]

    # ✅ Log to file
    logging.info(f"Input: {data.features}, Prediction: {prediction}")

    # ✅ Log to SQLite
    cursor.execute(
        "INSERT INTO predictions (input_data, prediction, timestamp) VALUES (?, ?, ?)",
        (str(data.features), float(prediction), datetime.now().isoformat())
    )
    conn.commit()

    return {"predicted_price": prediction}


@app.get("/logs")
def get_logs():
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    return {"recent_predictions": rows}


# ✅ Expose Prometheus metrics
Instrumentator().instrument(app).expose(app)
