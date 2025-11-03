# Diabetes Health Indicators Prediction

**Author:** Mahmoud M. Shoieb

## Project Overview
This repository contains an end-to-end classification project to predict the likelihood of diabetes using health indicators. The goal is to preprocess the dataset, explore and select features, train multiple classification models, tune hyperparameters, and evaluate model performance (accuracy, precision, recall, F1, ROC-AUC). The project demonstrates data cleaning, exploratory data analysis (EDA), classical ML modeling, and model evaluation.

**Dataset:** `diabetes_012_health_indicators_BRFSS2015.csv` (included in `/data`)

---

## Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Setup / Installation](#setup--installation)
- [How to run](#how-to-run)
- [Modeling & Results](#modeling--results)
- [My Contributions](#my-contributions)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Contact](#contact)

---

## Tech Stack
- Python 3.8+  
- Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`  
- Environment management: `venv` or `conda`  
- Jupyter Notebook / .py scripts for experiments

---

## Repository Structure
```
Diabetes-Health-Indicators-Prediction/
├─ data/
│  └─ diabetes_012_health_indicators_BRFSS2015.csv
├─ notebooks/
│  └─ diabetes_analysis.ipynb
├─ src/
│  ├─ preprocessing.py
│  ├─ modeling.py
│  └─ evaluate.py
├─ docs/
│  └─ Diabetes_Health_Indicators_Prediction_Report.pdf
├─ requirements.txt
├─ .gitignore
└─ README.md
```

---

## Setup / Installation

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/Diabetes-Health-Indicators-Prediction.git
cd Diabetes-Health-Indicators-Prediction
```

**2. Create virtual environment (recommended)**  
Using `venv`:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install requirements**
```bash
pip install -r requirements.txt
```
> If `requirements.txt` is not present, you can install the main libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## How to run

**Start Jupyter Notebook**  
```bash
jupyter notebook
# then open notebooks/diabetes_analysis.ipynb
```

**Or run scripts (example)**  
```bash
# Preprocess data
python src/preprocessing.py --input data/diabetes_012_health_indicators_BRFSS2015.csv --output data/processed.csv

# Train / evaluate (example)
python src/modeling.py --data data/processed.csv --model logistic
```

(Adjust script names & CLI args according to how you organize your `src/` files.)

---

## Modeling & Results
- Models trained: Logistic Regression, K-Nearest Neighbors, Decision Tree (and optionally GridSearchCV for tuning).  
- Evaluation metrics included: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC.  
- The notebook `notebooks/diabetes_analysis.ipynb` includes EDA plots, feature selection using `SelectKBest(f_classif)`, learning curves, and model comparison tables.

> See the `docs/Diabetes_Health_Indicators_Prediction_Report.pdf` for full details, experiments, graphs, and conclusions.

---

## My Contributions
I completed this project individually. Contributions include:
- Full dataset cleaning and preprocessing (missing values, encoding, scaling)
- Feature selection and engineering
- Implementation and hyperparameter tuning of classification models
- Model evaluation and visualization (ROC curves, confusion matrix)
- Writing the project report and notebook code

---

## Acknowledgements
- Dataset source: BRFSS (2015) (file included for convenience)
- No collaborators on this project — completed individually.

---

## License
This repository is available under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
Mahmoud Shoieb — (your email)  
Project available on GitHub: `https://github.com/<your-username>/Diabetes-Health-Indicators-Prediction`

