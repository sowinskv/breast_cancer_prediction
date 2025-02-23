import os
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from utils import load_processed_data, load_model
from visualization import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"dokładność: {accuracy:.4f}")
    print("raport klasyfikacji:")
    print(report)

    return y_pred, y_prob, cm, accuracy, report


if __name__ == "__main__":
    processed_dir = "../data/processed"
    models_dir = "../models"
    evaluation_dir = "../evaluation"

    # wczytanie danych i najlepszego modelu
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)
    model = load_model(models_dir, "best_model.pkl")

    # ewaluacja modelu
    y_pred, y_prob, cm, accuracy, report = evaluate_model(model, X_test, y_test)

    # wizualizacja
    os.makedirs(evaluation_dir, exist_ok=True)
    plot_confusion_matrix(cm, evaluation_dir)
    plot_roc_curve(y_test, y_prob, evaluation_dir)
    plot_precision_recall_curve(y_test, y_prob, evaluation_dir)

    # raport ewaluacji
    report_path = os.path.join(evaluation_dir, "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write(f"dokładność: {accuracy:.4f}\n")
        f.write("raport klasyfikacji:\n")
        f.write(report)
    print(f"raport zapisano: {report_path}")
