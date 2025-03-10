**Project Description:**  
The project aims to develop a classification model to predict whether a patient has breast cancer based on features extracted from X-ray images.

**Project Objective:**  
Select the best model for predicting breast cancer based on the aforementioned data.

**File Structure:**  
- **`data/`**: Directory containing data  
  - **`raw/`**: Raw data  
  - **`processed/`**: Processed data  
- **`models/`**: Saved models  
- **`src/`**: Source code  
  - **`data_preprocessing.py`**: Scripts for loading, cleaning, and processing data  
  - **`model_training.py`**: Scripts for training and evaluating models  
  - **`model_evaluation.py`**: Scripts for model evaluation  
  - **`visualization.py`**: Scripts for visualizing results  
  - **`utils.py`**: Utility functions for loading and saving data and models  
- **`evaluation/`**: Directory containing evaluation results  
- **`requirements.txt`**: File listing the project requirements

**How to Run:**  
1. Required Packages:  
   ```bash
   pip install -r requirements.txt
   ```  
2. Data Processing:
   ```bash
   python src/data_preprocessing.py
   ```  
3. Model Training: 
   ```bash
   python src/model_training.py
   ```  
4. Best Model Evaluation:
   ```bash
   python src/model_evaluation.py
   ```
