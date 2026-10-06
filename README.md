# 💼 Nettech Employee Salary Prediction

An end-to-end **Machine Learning salary prediction application** built with **Python, scikit-learn, and Streamlit**.

The application predicts an employee's estimated annual salary in INR based on factors such as **experience, education, skills, department, and age**. It also provides dataset exploration, model comparison, evaluation metrics, prediction visualizations, and feature importance analysis through an interactive web interface.

> **Note:** The included dataset is synthetically generated for project and demonstration purposes. It is not real employee salary data.

---

## 📌 Project Overview

Salary prediction is a regression problem where multiple employee attributes can influence compensation.

This project demonstrates a complete machine learning workflow:

```text
Employee Dataset
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Categorical Encoding
       ↓
Feature Scaling
       ↓
Train/Test Split
       ↓
Multiple Regression Models
       ↓
Model Evaluation
       ↓
Cross-Validation
       ↓
Best Model Selection
       ↓
Salary Prediction
       ↓
Interactive Streamlit Dashboard
```

---

## 🎯 Objectives

* Predict annual employee salary in INR.
* Analyze the relationship between experience and salary.
* Study the impact of education and department on salary.
* Convert employee skills into machine-learning features.
* Compare multiple regression algorithms.
* Evaluate models using R², MAE, RMSE, and 5-fold cross-validation.
* Automatically select the best-performing model.
* Provide an interactive interface for salary prediction.
* Visualize actual vs predicted salaries and prediction errors.
* Identify the most influential features affecting salary predictions.

---

## 🛠️ Technologies Used

| Technology   | Purpose                            |
| ------------ | ---------------------------------- |
| Python       | Core programming language          |
| Pandas       | Data manipulation and analysis     |
| NumPy        | Numerical operations               |
| Scikit-learn | Machine learning and preprocessing |
| Streamlit    | Interactive web application        |
| Matplotlib   | Data visualization                 |
| Joblib       | Model serialization and loading    |
| CSV          | Dataset storage                    |

---

## 🤖 Machine Learning Models

The project compares four regression algorithms:

### 1. Linear Regression

Used as a simple baseline model to establish a relationship between employee attributes and salary.

### 2. Ridge Regression

A regularized version of linear regression that helps reduce the effect of multicollinearity and overly large coefficients.

### 3. Random Forest Regressor

An ensemble tree-based model capable of capturing non-linear relationships between employee characteristics and salary.

### 4. Gradient Boosting Regressor

A boosting-based ensemble algorithm that builds models sequentially to improve prediction performance.

The models are evaluated on unseen test data, and **5-fold cross-validation R²** is used for selecting the final best model.

---

## 🧠 Feature Engineering

The `Skills` column contains multiple skills separated by semicolons, for example:

```text
Python;SQL;Machine Learning;Communication
```

Instead of treating this as a single text value, the project converts individual skills into binary features.

For example:

```text
skill_Python = 1
skill_SQL = 1
skill_Machine Learning = 1
skill_Excel = 0
```

A `Skill_Count` feature is also created to represent the total number of skills possessed by an employee.

### Features used

* Experience
* Age
* Education
* Department
* Individual skill indicators
* Skill count

Categorical variables such as **Education** and **Department** are transformed using **One-Hot Encoding**.

Numerical features are scaled for models that benefit from standardization.

---

## 📊 Model Evaluation

Each model is evaluated using:

### R² Score

Measures how much of the variation in salary is explained by the model.

Higher is better.

### MAE — Mean Absolute Error

Measures the average absolute difference between actual and predicted salary.

Lower is better.

### RMSE — Root Mean Squared Error

Measures prediction error while giving greater weight to larger errors.

Lower is better.

### 5-Fold Cross-Validation

The training data is divided into five folds to evaluate how consistently the model performs across different subsets of the data.

The model with the highest average cross-validation R² is selected as the final model.

---

## 📈 Application Features

The Streamlit application contains three main sections.

### 🎯 1. Predict Salary

Users can enter:

* Department
* Education
* Experience
* Age
* Skills

The application then provides:

* Predicted annual salary
* Approximate salary range based on model MAE
* Approximate monthly salary
* Salary comparison against employees in the selected department

---

### 📊 2. Data Explorer

The application provides:

* Dataset preview
* Number of employees
* Average salary
* Minimum salary
* Maximum salary
* Experience vs Salary scatter plot
* Average salary by department
* Average salary by education
* Salary distribution

---

### 📈 3. Model Performance

The model evaluation section displays:

* Model comparison table
* R² score
* MAE
* RMSE
* 5-fold cross-validation R²
* Best model
* Actual vs Predicted salary
* Prediction error/residual distribution
* Top factors influencing salary

---

## 📁 Project Structure

```text
nettech_salary/
│
├── app.py
├── ml.py
├── train.py
├── generate_data.py
├── requirements.txt
├── README.md
│
├── data/
│   └── employee_data.csv
│
└── models/
    └── salary_model.joblib
```

### File Description

| File                     | Description                                                                                            |
| ------------------------ | ------------------------------------------------------------------------------------------------------ |
| `app.py`                 | Streamlit application and interactive UI                                                               |
| `ml.py`                  | Data preprocessing, feature engineering, model training, evaluation, prediction and feature importance |
| `train.py`               | Command-line script for training and comparing models                                                  |
| `generate_data.py`       | Generates the synthetic employee salary dataset                                                        |
| `requirements.txt`       | Required Python packages                                                                               |
| `data/employee_data.csv` | Employee salary dataset                                                                                |
| `models/`                | Stores the trained model                                                                               |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/nettech-employee-salary-prediction.git
```

### 2. Navigate to the project

```bash
cd nettech-employee-salary-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🧪 Train the Model Manually

To generate a new synthetic dataset:

```bash
python generate_data.py
```

To train and compare all models:

```bash
python train.py
```

The training script displays the performance of each model and saves the selected best model to:

```text
models/salary_model.joblib
```

---

## 📋 Dataset

The dataset contains the following columns:

| Column       | Description                         |
| ------------ | ----------------------------------- |
| `Experience` | Years of professional experience    |
| `Education`  | Highest education level             |
| `Skills`     | Semicolon-separated employee skills |
| `Department` | Employee department/role            |
| `Age`        | Employee age                        |
| `Salary`     | Annual salary in INR                |

The included dataset contains **200 employee records** and is generated using predefined salary relationships and controlled random variation.

---

## 🔍 Example Prediction Workflow

A user can enter:

```text
Department: Technical Trainers
Education: Bachelors
Experience: 5 years
Age: 27
Skills: Python, SQL, Machine Learning
```

The application processes these inputs through the same feature-engineering and preprocessing pipeline used during model training and returns an estimated annual salary.

---

## 📊 Visualizations

The application includes visualizations such as:

* Experience vs Salary
* Average Salary by Department
* Average Salary by Education
* Salary Distribution
* Actual vs Predicted Salary
* Prediction Error Distribution
* Feature Importance
* Prediction vs Department Salary Distribution

These visualizations help understand both the dataset and model behavior.

---

## 🔄 Model Pipeline

The complete ML pipeline can be summarized as:

```text
Raw Employee Data
       ↓
Remove Missing Salary Records
       ↓
Remove Duplicate Records
       ↓
Feature Engineering
       ↓
Skill Extraction
       ↓
Skill Count
       ↓
One-Hot Encoding
       ↓
Feature Scaling
       ↓
Train/Test Split
       ↓
Train 4 Regression Models
       ↓
Evaluate R² / MAE / RMSE
       ↓
5-Fold Cross-Validation
       ↓
Select Best Model
       ↓
Save Model with Joblib
       ↓
Streamlit Prediction App
```

---

## ⚠️ Limitations

* The included dataset is synthetic and should not be used to make real-world compensation decisions.
* Salary prediction depends heavily on the quality and representativeness of the training data.
* The model captures patterns present in the generated dataset and may not generalize to real employee populations.
* Predictions should be treated as estimates rather than guaranteed salary values.
* Additional real-world factors such as company size, location, industry, performance, seniority, and market conditions are not included.

---

## 🔮 Future Improvements

Potential improvements include:

* Train on real-world salary datasets.
* Add location and company-size features.
* Include job title and seniority level.
* Add more advanced models such as XGBoost or LightGBM.
* Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
* Add explainable AI using SHAP.
* Add downloadable prediction reports.
* Deploy the application publicly using Streamlit Community Cloud.
* Add authentication and database integration.
* Track prediction history.

---

## 👩‍💻 Skills Demonstrated

This project demonstrates practical experience in:

* Python
* Machine Learning
* Regression
* Feature Engineering
* Data Preprocessing
* One-Hot Encoding
* Feature Scaling
* Cross-Validation
* Model Evaluation
* Data Visualization
* Streamlit
* Scikit-learn
* Pandas
* NumPy
* Model Deployment Concepts

---

## 📜 License

This project is intended for educational and portfolio purposes.
