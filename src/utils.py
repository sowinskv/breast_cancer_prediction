import os
import pandas as pd
import joblib


def load_processed_data(processed_dir):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv")).squeeze()

    return X_train, X_test, y_train, y_test


def save_model(model, models_dir, filename):
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, filename)
    joblib.dump(model, model_path)
    print(f"zapisano model: {model_path}")


def load_model(models_dir, filename):
    model_path = os.path.join(models_dir, filename)
    model = joblib.load(model_path)
    print(f"wczytano model: {model_path}")
    return model
