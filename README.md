# Alzheimers Prediction

## Overview
This project predicts Alzheimer’s diagnosis using Alzheimer dataset available on Kaggle, by using patient data, including demographic details, lifestyle factors, medical history, clinical measurements, and cognitive assessments. The goal is to identify key predictors and build accurate machine learning models.

Features
The dataset contains 33 features after preprocessing, grouped into categories:

1. Demographic Details: Age, Gender, Ethnicity, Education Level.
2. Lifestyle Factors: BMI, Smoking, Alcohol Consumption, Physical Activity, Diet Quality, Sleep Quality.
3. Medical History: Family history of Alzheimer’s, presence of chronic conditions like cardiovascular disease, diabetes, depression, and hypertension.
4. Clinical Measurements: Blood pressure, cholesterol levels, and triglycerides. 
5. Cognitive and Functional Assessments: Mini-Mental State Examination (MMSE), functional assessment scores, and behavioral symptoms. 
6. Symptoms: Confusion, forgetfulness, disorientation, and task difficulties. 
7. Diagnosis: The target variable (binary: Alzheimer’s or not).
## Workflow
### Data Preprocessing:
- Dropped non-predictive columns: PatientID and DoctorInCharge. 
- Handled missing values and duplicates.

- Categorized features into numerical and categorical.

### Exploratory Data Analysis:
- isualized distributions of key features.

- Assessed feature relationships with the target variable.
### Modeling:
- Implemented Logistic Regression and Random Forest models.

- Used cross-validation for robust evaluation.

- Compared models using metrics like accuracy, F1-score, and ROC-AUC.

## Key Results
- Age, MMSE, and CDR were the strongest predictors.
- Random Forest provided higher accuracy and ROC-AUC, while Logistic Regression offered interpretability.
## Usage
Install dependencies from requirements.txt

```shell
pip install -r requirements.txt
```
Open and run AlzheimersNotebook.ipynb in Jupyter Notebook.
Replace the dataset with a properly formatted file if needed.
## Files
- AlzheimersNotebook.ipynb: Main analysis notebook.
-  Dataset: Ensure compatibility with the original structure.
## Source
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset/data