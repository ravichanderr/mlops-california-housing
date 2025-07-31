import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def load_and_save_california_data(output_path="data/raw/california.csv"):
    """Loads the California Housing dataset and saves it as a CSV."""
    
    # Load dataset
    print("📥 Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset saved at: {output_path}")
    print(f"🔍 Shape: {df.shape}")

if __name__ == "__main__":
    load_and_save_california_data()
