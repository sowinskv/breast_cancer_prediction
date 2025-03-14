**description:**  
this project aims to develop a classification model to predict whether a patient has breast cancer based on features extracted from X-ray images

**objective:**  
select the best model for predicting breast cancer based on the aforementioned data

**structure::**  
- **`data/`**: dir for data
  - **`raw/`**: raw data  
  - **`processed/`**: processed data  
- **`models/`**: saved models  
- **`src/`**: source code  
  - **`data_preprocessing.py`**: scripts for loading, cleaning, and processing data  
  - **`model_training.py`**: training and evaluating models  
  - **`model_evaluation.py`**: model evaluation  
  - **`visualization.py`**: visualizing results  
  - **`utils.py`**: utility functions
- **`evaluation/`**: dir containing evaluation results  
- **`requirements.txt`**: project requirements

**run:**  
1. install required packages:  
   ```bash
   pip install -r requirements.txt
   ```  
2. process data:
   ```bash
   python src/data_preprocessing.py
   ```  
3. train models: 
   ```bash
   python src/model_training.py
   ```  
4. model evaluation:
   ```bash
   python src/model_evaluation.py
   ```
