import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="CreditWise", page_icon="💳", layout="wide")

# Minimal but fancy touch — just 6 lines of CSS
st.markdown("""
<style>
    .big-title { font-size: 2.4rem; font-weight: 800; color: #4f8ef7; letter-spacing: -1px; }
    .sub       { color: #888; font-size: 0.9rem; margin-top: -10px; margin-bottom: 1rem; }
    .approve   { background: #e6f9f0; border-left: 5px solid #22c55e; padding: 1rem; border-radius: 8px; }
    .reject    { background: #fef2f2; border-left: 5px solid #ef4444; padding: 1rem; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model (cached) ──────────────────────────────────────────────────────

@st.cache_data
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    df2 = df.copy().drop("Applicant_ID", axis=1, errors="ignore")

    # Drop rows with any NaN
    df2 = df2.dropna()

    # Map target to binary
    df2["Loan_Approved"] = df2["Loan_Approved"].map({"Yes": 1, "No": 0})

    le = LabelEncoder()
    enc_map = {}
    for col in df2.select_dtypes("object").columns:
        df2[col] = le.fit_transform(df2[col].astype(str))
        enc_map[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df2.drop("Loan_Approved", axis=1)
    y = df2["Loan_Approved"]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        yp    = m.predict(X_test)
        yprob = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, yprob)
        results[name] = {
            "model": m, "accuracy": accuracy_score(y_test, yp),
            "roc_auc": roc_auc_score(y_test, yprob),
            "cm": confusion_matrix(y_test, yp), "fpr": fpr, "tpr": tpr,
        }
    return df, X, y, scaler, enc_map, results


try:
    df, X, y, scaler, enc_map, results = load_and_train()
    loaded = True
except Exception as e:
    st.error(f"Could not load data: {e}")
    loaded = False


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 💳 CreditWise")
    st.caption("ML Loan Approval System")
    st.divider()
    page = st.radio("Go to", ["Overview", "EDA", "Models", "Predict"])
    if loaded:
        st.divider()
        st.metric("Records", f"{len(df):,}")
        st.metric("Features", df.shape[1])


if not loaded:
    st.stop()


# ── Overview ───────────────────────────────────────────────────────────────────

if page == "Overview":
    st.markdown('<div class="big-title">CreditWise Loan System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">ML-powered credit decision intelligence</div>', unsafe_allow_html=True)
    st.divider()

    approved = (df["Loan_Approved"] == "Y").sum() if df["Loan_Approved"].dtype == object \
               else (df["Loan_Approved"] == 1).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Applicants", f"{len(df):,}")
    c2.metric("Approved", f"{approved:,}")
    c3.metric("Approval Rate", f"{approved/len(df)*100:.1f}%")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ── EDA ────────────────────────────────────────────────────────────────────────

elif page == "EDA":
    st.markdown('<div class="big-title">Exploratory Analysis</div>', unsafe_allow_html=True)
    st.divider()

    tab1, tab2, tab3 = st.tabs(["Distributions", "Correlation", "Feature vs Approval"])

    with tab1:
        col = st.selectbox("Feature", df.select_dtypes(include=np.number).columns)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30, color="#4f8ef7", edgecolor="white")
        ax.set_xlabel(col); ax.set_ylabel("Count")
        st.pyplot(fig); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.select_dtypes(include=np.number).corr(),
                    annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
        st.pyplot(fig); plt.close()

    with tab3:
        col2 = st.selectbox("Numeric feature", df.select_dtypes(include=np.number).columns, key="f2")
        fig, ax = plt.subplots()
        for label, color in zip(df["Loan_Approved"].unique(), ["#22c55e", "#ef4444"]):
            ax.hist(df[df["Loan_Approved"] == label][col2].dropna(),
                    bins=25, alpha=0.6, color=color, label=f"Approved: {label}")
        ax.legend(); ax.set_xlabel(col2)
        st.pyplot(fig); plt.close()


# ── Models ─────────────────────────────────────────────────────────────────────

elif page == "Models":
    st.markdown('<div class="big-title">Model Results</div>', unsafe_allow_html=True)
    st.divider()

    comp = pd.DataFrame([
        {"Model": n, "Accuracy": f"{r['accuracy']*100:.2f}%", "ROC-AUC": f"{r['roc_auc']:.4f}"}
        for n, r in results.items()
    ])
    st.dataframe(comp, use_container_width=True, hide_index=True)

    chosen = st.selectbox("Inspect model →", list(results.keys()))
    r = results[chosen]

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, linewidths=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig); plt.close()

    with c2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(r["fpr"], r["tpr"], color="#4f8ef7", lw=2, label=f"AUC = {r['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.fill_between(r["fpr"], r["tpr"], alpha=0.1, color="#4f8ef7")
        ax.legend(); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        st.pyplot(fig); plt.close()

    if chosen in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Feature Importance")
        fi = pd.Series(r["model"].feature_importances_, index=X.columns)\
               .sort_values(ascending=True).tail(10)
        fig, ax = plt.subplots()
        fi.plot(kind="barh", ax=ax, color="#4f8ef7")
        st.pyplot(fig); plt.close()


# ── Predict ────────────────────────────────────────────────────────────────────

elif page == "Predict":
    st.markdown('<div class="big-title">Predict Loan</div>', unsafe_allow_html=True)
    st.divider()

    best = max(results, key=lambda k: results[k]["roc_auc"])
    model_choice = st.selectbox("Model", list(results.keys()),
                                index=list(results.keys()).index(best))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Personal")
        applicant_income   = st.number_input("Applicant Income",   0, value=5000,   step=500)
        coapplicant_income = st.number_input("Co-applicant Income", 0, value=0,      step=500)
        age                = st.number_input("Age", 18, 75, 35)
        dependents         = st.number_input("Dependents", 0, 10, 0)
        gender             = st.selectbox("Gender", ["Male", "Female"])
        marital_status     = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
        education_level    = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])

    with c2:
        st.subheader("Loan Details")
        loan_amount        = st.number_input("Loan Amount", 1000, value=150000, step=5000)
        loan_term          = st.number_input("Term (months)", 12, 360, 120, step=12)
        loan_purpose       = st.selectbox("Purpose", ["Home", "Car", "Education", "Business", "Personal"])
        property_area      = st.selectbox("Property Area", ["Urban", "Suburban", "Rural"])
        employer_category  = st.selectbox("Employer", ["Government", "Private", "Self-Employed", "NGO"])
        employment_status  = st.selectbox("Employment", ["Employed", "Self-Employed", "Unemployed"])

    with c3:
        st.subheader("Financials")
        credit_score       = st.slider("Credit Score", 300, 850, 680)
        dti_ratio          = st.slider("DTI Ratio", 0.0, 1.0, 0.35, 0.01)
        savings            = st.number_input("Savings", 0, value=10000, step=1000)
        collateral_value   = st.number_input("Collateral Value", 0, value=50000, step=5000)
        existing_loans     = st.number_input("Existing Loans", 0, 20, 1)

    if st.button("Predict", use_container_width=True, type="primary"):
        inp = {
            "Applicant_Income": applicant_income, "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status, "Age": age, "Marital_Status": marital_status,
            "Dependents": dependents, "Credit_Score": credit_score, "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio, "Savings": savings, "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount, "Loan_Term": loan_term, "Loan_Purpose": loan_purpose,
            "Property_Area": property_area, "Education_Level": education_level,
            "Gender": gender, "Employer_Category": employer_category,
        }
        inp_df = pd.DataFrame([inp])
        for col in inp_df.select_dtypes("object").columns:
            if col in enc_map:
                inp_df[col] = enc_map[col].get(str(inp_df[col].iloc[0]), 0)
        for col in X.columns:
            if col not in inp_df.columns:
                inp_df[col] = 0
        inp_df = inp_df[X.columns]

        prob     = results[model_choice]["model"].predict_proba(scaler.transform(inp_df))[0]
        approved = prob[1] >= 0.5

        st.divider()
        if approved:
            st.markdown(f'<div class="approve"><h3>✅ Loan Approved</h3><p>Confidence: <strong>{prob[1]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="reject"><h3>❌ Loan Rejected</h3><p>Confidence: <strong>{prob[0]*100:.1f}%</strong></p></div>', unsafe_allow_html=True)

        st.progress(float(prob[1]), text=f"Approval probability: {prob[1]*100:.1f}%")
