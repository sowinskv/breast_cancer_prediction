opis projektu:

projekt ma na celu opracowanie modelu klasyfikacyjnego do prognozowania czy pacjent jest chory na raka piersi, na podstawie cech wyciągniętych ze zdjęć rentgenowskich

cel projektu:

wybranie najlepszego modelu do prognozowania raka piersi na podstawie w.w. danych

struktura plików:
- `data/`: katalog z danymi
  - `raw/`: surowe dane
  - `processed/`: przetworzone dane
- `models/`: zapisane modele
- `src/`: kod źródłowy
  - `data_preprocessing.py`: skrypty do wczytywania, czyszczenia i przetwarzania danych
  - `model_training.py`: skrypty do trenowania i oceny modeli
  - `model_evaluation.py`: skrypty do ewaluacji modeli
  - `visualization.py`: skrypty do wizualizacji wyników
  - `utils.py`: pomocnicze funkcje do wczytywania i zapisywania danych oraz modeli
- `evaluation/`: katalog z wynikami ewaluacji
- `requirements.txt`: plik z wymaganiami

sposób uruchamiania:
1. wymagane pakiety:
   ```bash
   pip install -r requirements.txt
   ```
2. przetwarzanie danych:
   ```bash
   python src/data_preprocessing.py
   ```
3. trenowanie modeli:
    ```bash
    python src/model_training.py
    ```
4. ewaluacja najlepszego modelu:
    ```bash
    python src/model_evaluation.py
   ```
