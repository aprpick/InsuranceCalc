# Insurance Cost Prediction Model

This project builds and deploys a machine learning model to predict U.S. health insurance costs based on user demographic and behavioral data. It uses a linear regression model trained and tested via Azure Machine Learning, with deployment support for both local and cloud environments.

---

## 💻 Project Structure

```
Azure_LG/
└── Model Training/
    ├── Dependencies/
    │   ├── config.json         # Azure ML workspace config
    │   ├── requirements.txt    # Required packages for training/inference
    │   └── score.py            # Scoring script for deployment
    └── Scripts/
        ├── 1_Feature_Engineering.py   # Preprocessing and feature setup
        └── 2_Model_Train.py           # Training script (with R² logging)
```

---

## 🧠 Features

- Linear regression model with Azure ML integration
- Modular pipeline with separated training and scoring
- Designed for deployment via Azure or locally with Docker
- Cleaned and engineered features from Kaggle insurance dataset

---

## ⚙️ Requirements

Install packages via pip:

```bash
pip install -r Azure_LG/Model\ Training/Dependencies/requirements.txt
```

---

## 🚀 Training the Model

You can train the model locally or push it to Azure ML for remote training:

**Local training:**
```bash
python Azure_LG/Model\ Training/Scripts/2_Model_Train.py
```

**Azure ML submission:**
Configure `config.json` and use the Azure ML SDK to submit.

---

## 📁 Data Note

Large data files (CSV, PKL) have been excluded from this repo to keep it lightweight.
You can use a dataset such as:
[Kaggle: Insurance Data for Machine Learning](https://www.kaggle.com/datasets/sridharstreaks/insurance-data-for-machine-learning)

---

## 👤 Author

**Adam Prpick**  
Petroleum Engineer in Training (EIT) with APEGA  
[GitHub](https://github.com/aprpick) | [LinkedIn](https://www.linkedin.com/in/a-p-0b319520b/)
