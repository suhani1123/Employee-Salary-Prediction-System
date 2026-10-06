import os
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import ml

st.set_page_config(page_title="NetTech Employee Salary Predictor", page_icon="💼", layout="wide")


@st.cache_data
def get_data():
    if not os.path.exists(ml.DATA_PATH):
        import generate_data
        generate_data.generate().to_csv(ml.DATA_PATH, index=False)
    return ml.load_data()


@st.cache_resource
def get_bundle():
    # Load the saved model; if it is missing or was saved with different library
    # versions (pickle incompatibility), simply retrain it on this machine.
    try:
        if os.path.exists(ml.MODEL_PATH):
            return ml.load_bundle()
    except Exception:
        pass
    return ml.train(get_data())


def inr(x):
    return f"₹{x:,.0f}"


df, bundle = get_data(), get_bundle()

st.title("💼 NetTech Employee Salary Prediction")

with st.sidebar:
    st.header("About")
    st.write(f"**Best model:** {bundle['best_name']}")
    st.write(f"**Employees in dataset:** {len(df)}")
    st.write(f"**Train / Test split:** {bundle['n_train']} / {bundle['n_test']}")
    if st.button("🔄 Retrain model"):
        st.cache_resource.clear()
        bundle = ml.train(get_data())
        st.success("Model retrained!")
        st.rerun()

tab_pred, tab_data, tab_model = st.tabs(["🎯 Predict Salary", "📊 Data Explorer", "📈 Model Performance"])


with tab_pred:
    st.subheader("Enter Employee Information")
    c1, c2 = st.columns(2)
    with c1:
        department = st.selectbox("Department", ml.DEPARTMENTS)
        education = st.selectbox("Education", ml.EDUCATION, index=1)
        experience = st.slider("Experience (years)", 0.0, 30.0, 3.0, 0.5)
    with c2:
        age = st.slider("Age", 20, 60, max(22, int(21 + experience)))
        skills = st.multiselect("Skills", ml.ALL_SKILLS, default=["Communication"])

    if age - experience < 17:
        st.warning("Experience looks too high for this age; the prediction may be unreliable.")

    if st.button("Predict Salary", type="primary"):
        pred = ml.predict(bundle, experience, age, education, department, skills)
        mae = bundle["results"][bundle["best_name"]]["MAE"]
        st.metric("Predicted annual salary", inr(pred))
        st.write(f"Typical range (± average model error): **{inr(max(pred - mae, 0))} – {inr(pred + mae)}**")
        st.write(f"Approx. monthly: **{inr(pred / 12)}**")

        peers = df[df["Department"] == department]["Salary"]
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(peers, bins=25, color="#9ecae1", edgecolor="white")
        ax.axvline(pred, color="crimson", lw=2, label="Predicted")
        ax.set_title(f"Prediction vs {department} salaries")
        ax.set_xlabel("Annual salary (₹)"); ax.legend()
        st.pyplot(fig)


with tab_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50))
    a, b, c, d = st.columns(4)
    a.metric("Employees", len(df)); b.metric("Avg Salary", inr(df.Salary.mean()))
    c.metric("Min Salary", inr(df.Salary.min())); d.metric("Max Salary", inr(df.Salary.max()))

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(df.Experience, df.Salary, s=10, alpha=.5, c="#3182bd")
        ax.set_xlabel("Experience (years)"); ax.set_ylabel("Salary (₹)"); ax.set_title("Experience vs Salary")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        g = df.groupby("Department").Salary.mean().sort_values()
        ax.barh(g.index, g.values, color="#31a354")
        ax.set_title("Average Salary by Department"); ax.set_xlabel("Salary (₹)")
        st.pyplot(fig)
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots()
        g = df.groupby("Education").Salary.mean().reindex(ml.EDUCATION)
        ax.bar(g.index, g.values, color="#e6550d")
        ax.set_title("Average Salary by Education"); ax.set_ylabel("Salary (₹)")
        st.pyplot(fig)
    with col4:
        fig, ax = plt.subplots()
        ax.hist(df.Salary, bins=30, color="#756bb1", edgecolor="white")
        ax.set_title("Salary Distribution"); ax.set_xlabel("Salary (₹)")
        st.pyplot(fig)

# ---------------- Model Performance ----------------
with tab_model:
    st.subheader("Model Comparison (on unseen test data)")
    res = pd.DataFrame(bundle["results"]).T.rename(columns={"CV_R2": "CV R² (5-fold)"})
    st.dataframe(res.style.format({"R2": "{:.3f}", "CV R² (5-fold)": "{:.3f}", "MAE": "₹{:,.0f}", "RMSE": "₹{:,.0f}"})
                 .highlight_max(subset=["R2", "CV R² (5-fold)"], color="#c7e9c0")
                 .highlight_min(subset=["MAE", "RMSE"], color="#c7e9c0"))
    best = bundle["results"][bundle["best_name"]]
    st.success(f"Best model: **{bundle['best_name']}** — explains {best['R2']*100:.1f}% of salary variation, "
               f"average error {inr(best['MAE'])}.")

    t = bundle["test"]
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(t.Actual, t.Predicted, s=12, alpha=.6)
        lim = [t.Actual.min(), t.Actual.max()]
        ax.plot(lim, lim, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual (₹)"); ax.set_ylabel("Predicted (₹)"); ax.set_title("Actual vs Predicted"); ax.legend()
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.hist(t.Predicted - t.Actual, bins=25, color="#fd8d3c", edgecolor="white")
        ax.axvline(0, color="k", lw=1)
        ax.set_title("Prediction errors (residuals)"); ax.set_xlabel("Predicted − Actual (₹)")
        st.pyplot(fig)

    fi = ml.feature_importance(bundle).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(fi.index, fi.values, color="#6baed6")
    ax.set_title("Top Factors influencing Salary"); ax.set_xlabel("Relative importance")
    st.pyplot(fig)
