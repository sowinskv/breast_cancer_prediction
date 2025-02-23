import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_processed_data, save_model
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier


def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_accuracy = 0
    results = []

    for name, model in models.items():
        print(f"trenowanie: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, acc))
        print(f"dokladnosc dla {name}: {acc:.4f}")

        if acc > best_accuracy:
            best_model = model
            best_accuracy = acc

    return best_model, results


if __name__ == "__main__":
    processed_dir = "../data/processed"
    models_dir = "../models"

    # load data
    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    print(f"y_train: {pd.Series(y_train).isnull().sum()} NaN values")
    print(f"y_test: {pd.Series(y_test).isnull().sum()} NaN values")

    # usuwanie NaN (jesli sa)
    y_train = y_train[~pd.isnull(y_train)]
    y_test = y_test[~pd.isnull(y_test)]

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "NeuralNetwork": MLPClassifier(random_state=42),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "Bagging": BaggingClassifier(random_state=42),
        "Boosting": AdaBoostClassifier(random_state=42)
    }

    best_model, results = train_and_evaluate(models, X_train, y_train, X_test, y_test)

    save_model(best_model, models_dir, "best_model.pkl")

    print("\nporownanie modeli:")
    for name, acc in results:
        print(f"{name}: {acc:.4f}")
