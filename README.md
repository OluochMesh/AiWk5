# 🏥 Hospital Readmission Prediction Using Machine Learning

This project predicts whether a patient is likely to be **readmitted within 30 days** using the **Diabetes 130-US Hospitals Dataset**.  
It applies real-world data preprocessing, class-imbalance handling, and a tuned **XGBoost classifier**, then exposes the model via an interactive **Streamlit app**.

---

## 📌 Project Overview

Hospital readmissions are costly, risky, and heavily monitored.  
This project builds a machine learning pipeline to:

✅ Clean and preprocess clinical records  
✅ Encode categorical + numerical features  
✅ Handle severe class imbalance  
✅ Train an XGBoost model  
✅ Evaluate its performance  
✅ Provide an easy-to-use prediction UI (Streamlit)

The final model predicts whether a patient will be **readmitted within 30 days** (1 = Yes, 0 = No).

---

## 📁 Dataset Description

**Dataset:** *Diabetes 130-US Hospitals for 1999–2008*  
**Source:** UCI Machine Learning Repository  

It contains more than **100,000 patient encounters**, including:

- Patient demographics  
- Diagnoses  
- Lab results  
- Medications  
- Hospital stay details  
- Readmission outcome (`<30`, `>30`, or `NO`)  

For this assignment, the target variable was converted to:

- **1 → Readmitted within 30 days**
- **0 → Not readmitted**

---

## 🧹 Data Preprocessing Steps

1. Replace missing placeholders `"?"` with `NaN`  
2. Drop irrelevant columns:
   - `weight`
   - `patient_nbr`
3. Convert the target column:
   ```python
   df["readmitted"] = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)

Separate features (X) and labels (y)

Build a preprocessing pipeline:

SimpleImputer for missing values

StandardScaler for numeric features

OneHotEncoder for categorical features

⚖️ Handling Class Imbalance

The dataset is extremely imbalanced:

88% = Not readmitted

12% = Readmitted

To fix this, the project applies:

✅ SMOTE oversampling (after encoding)
✅ scale_pos_weight in XGBoost
✅ Tuned hyperparameters for better recall

This dramatically improved model sensitivity.

🤖 Model Used — XGBoost Classifier
Key hyperparameters:

XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10,
    eval_metric="logloss",
    random_state=42
)

✅ Model Performance
Metric	Score
Accuracy	~0.50
Precision	~0.15
Recall	~0.76 ✅
F1-Score	~0.25
✅ Why low accuracy is acceptable here?

Because the dataset is heavily imbalanced, accuracy is misleading.
A model that always predicts "NO" would score 88% accuracy but be useless.

Recall is the main goal — catch as many risky patients as possible.
Your model’s recall of 0.76 is very strong for this dataset.

📸 Project Screenshots
✅ 1. Streamlit Home

(Insert image here)
![Streamlit Home](images/streamlit_home.png)

✅ 2. Prediction Form

(Insert image here)
![Prediction Page](images/prediction_form.png)

✅ 3. Prediction Result

(Insert image here)
![Prediction Output](images/prediction_output.png)

You can upload screenshots later — placeholders are already included.

🔧 How to Run the Project
✅ 1. Install dependencies
pip install -r requirements.txt

✅ 2. Train the model
python src/model.py


This will generate:

models/trained_model.pkl

✅ 3. Run Streamlit

Make sure you're inside the streamlit_app folder:

cd streamlit_app
streamlit run app_streamlit.py

📂 Project Structure
AiWk5/
│── data/
│   └── readmission.csv
│
│── models/
│   └── trained_model.pkl
│
│── src/
│   ├── preprocess.py
│   └── model.py
│
│── streamlit_app/
│   └── app_streamlit.py
│
│── README.md
│── requirements.txt

✅ Features Implemented

✅ End-to-end ML pipeline

✅ Clean and structured code

✅ Well-commented Python scripts

✅ Automatic preprocessing

✅ XGBoost classifier

✅ Class imbalance handling

✅ Streamlit UI

✅ Model saving / loading

📝 License

This project is for academic use under the course ML Assignment (Part 2: Case Study).

👨‍💻 Author

Meshack Odhiambo Oluoch
Bachelor of Information Technology
Masinde Muliro University of Science & Technology