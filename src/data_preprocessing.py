import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif


def select_features(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)


def load_data(data_path):
    column_names = [
        "ID", "Outcome", "Time", "Radius_mean", "Texture_mean", "Perimeter_mean",
        "Area_mean", "Smoothness_mean", "Compactness_mean", "Concavity_mean",
        "Concave_points_mean", "Symmetry_mean", "Fractal_dimension_mean",
        "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
        "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se",
        "Fractal_dimension_se", "Radius_worst", "Texture_worst", "Perimeter_worst",
        "Area_worst", "Smoothness_worst", "Compactness_worst", "Concavity_worst",
        "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst",
        "Tumor_size", "Lymph_node_status"
    ]

    with open(data_path, "r") as f:
        raw_data = f.readlines()[:5]
        print("1. wiersze surowych danych:")
        for line in raw_data:
            print(line.strip())

    # wczytanie danych
    data = pd.read_csv(data_path, header=None, names=column_names, sep=",", na_values=["?"])

    print("\ndane (pierwsze 5 wierszy):")
    print(data.head())
    print("\nliczba kolumn:", len(data.columns))

    return data


def clean_data(data):
    """
    - usuwanie niepotrzebnych kolumn
    - mapowanie etykiet na wartości numeryczne
    """
    data = data.drop(columns=["ID"])

    # sprawdzenie brakujących wartości
    print("Liczba brakujących wartości przed czyszczeniem:")
    print(data.isnull().sum())

    # konwersja 'Outcome' na string, jeśli nie jest stringiem
    if not pd.api.types.is_string_dtype(data["Outcome"]):
        print("Konwersja kolumny 'Outcome' na string...")
        data["Outcome"] = data["Outcome"].astype(str)

    # czyszczenie scalonych wartosci
    data["Outcome"] = data["Outcome"].str.strip()
    data["Outcome"] = data["Outcome"].str.extract(r'([RN])')[0]  # Ekstrakcja liter 'R' lub 'N'

    print("przed mapowaniem:")
    print(data["Outcome"].value_counts())

    # mapowanie 'Outcome' na wartości numeryczne
    data["Outcome"] = data["Outcome"].map({"R": 1, "N": 0})

    print("\npo mapowaniu:")
    print(data["Outcome"].value_counts())

    data = data.dropna()

    print("\ndane po czyszczeniu:")
    print(data.head())
    print(f"no. wierszy po czyszczeniu: {data.shape[0]}")
    return data


def preprocess_features(data):
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    # skalowanie
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame(X_train).to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    pd.DataFrame(X_test).to_csv(os.path.join(output_dir, "X_test.csv"), index=False)

    # konwersja na pandas i zapis
    y_train = pd.Series(y_train, name="Outcome")
    y_test = pd.Series(y_test, name="Outcome")
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


if __name__ == "__main__":
    data_path = "../data/raw/wpbc.data"
    output_path = "../data/processed"

    # wczytanie
    data = load_data(data_path)
    print("dane:")
    print(data.head())

    data_cleaned = clean_data(data)
    print("\npo czyszczeniu:")
    print(data_cleaned.head())

    # cechy i etykiety
    X, y = preprocess_features(data_cleaned)
    X_selected, selected_features = select_features(X, y, k=10)
    print("\ncechy + skalowanie:")
    print(X[:5])

    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(y_test.to_csv)

    # zapis danych przetworzonych
    save_processed_data(X_train, X_test, y_train, y_test, output_path)
