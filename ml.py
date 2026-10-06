import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data/employee_data.csv"
MODEL_PATH = "models/salary_model.joblib"

DEPARTMENTS = ["Technical Trainers", "Finance & Operations", "Business Development Executive",
               "HR", "Placement Coordinator", "Sales & Marketing"]
EDUCATION = ["Diploma", "Bachelors", "Masters", "MBA", "PhD"]
ALL_SKILLS = ["Python", "Java", "SQL", "Machine Learning", "Web Development", "Excel", "Accounting",
              "Tally", "Sales", "CRM", "Negotiation", "Communication", "Recruitment", "Payroll",
              "Networking", "Digital Marketing", "SEO"]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df.dropna(subset=["Salary"]).drop_duplicates().reset_index(drop=True)


def engineer_features(df):
    
    X = df[["Experience", "Age", "Education", "Department"]].copy()
    lists = df["Skills"].fillna("").apply(lambda s: [x.strip() for x in str(s).split(";") if x.strip()])
    for skill in ALL_SKILLS:
        X[f"skill_{skill}"] = lists.apply(lambda l: int(skill in l))
    X["Skill_Count"] = lists.apply(len)
    return X


def build_pipeline(model, scale=False):
    cat = ["Education", "Department"]
    num = [c for c in engineer_features(pd.DataFrame({"Experience": [0], "Age": [0], "Education": [""],
           "Department": [""], "Skills": [""]})).columns if c not in cat]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler() if scale else "passthrough", num),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def train(df):
    X, y = engineer_features(df), df["Salary"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "Linear Regression": build_pipeline(LinearRegression(), scale=True),
        "Ridge Regression": build_pipeline(Ridge(alpha=1.0), scale=True),
        "Random Forest": build_pipeline(RandomForestRegressor(n_estimators=200, random_state=42)),
        "Gradient Boosting": build_pipeline(GradientBoostingRegressor(random_state=42)),
    }
    results, fitted = {}, {}
    for name, pipe in candidates.items():
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "R2": r2_score(y_te, pred),
            "MAE": mean_absolute_error(y_te, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
            "CV_R2": float(cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2").mean()),
        }
        fitted[name] = pipe
    best = max(results, key=lambda k: results[k]["CV_R2"])
    pred = fitted[best].predict(X_te)
    bundle = {
        "model": fitted[best], "best_name": best, "results": results,
        "test": pd.DataFrame({"Actual": np.asarray(y_te, dtype=float), "Predicted": np.asarray(pred, dtype=float)}),
        "n_train": len(X_tr), "n_test": len(X_te),
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_bundle():
    return joblib.load(MODEL_PATH)


def predict(bundle, experience, age, education, department, skills):
    row = pd.DataFrame([{"Experience": experience, "Age": age, "Education": education,
                         "Department": department, "Skills": ";".join(skills)}])
    return float(bundle["model"].predict(engineer_features(row))[0])


def feature_importance(bundle):
    pipe = bundle["model"]
    names = pipe.named_steps["prep"].get_feature_names_out()
    m = pipe.named_steps["model"]
    vals = m.feature_importances_ if hasattr(m, "feature_importances_") else np.abs(m.coef_)
    s = pd.Series(vals, index=[n.split("__", 1)[1] for n in names]).sort_values(ascending=False)
    return s / s.sum()
