import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
    }
    .metric-card h2 { color: #4fc3f7; margin: 0; font-size: 2rem; }
    .metric-card p  { color: #aaa; margin: 0; font-size: 0.85rem; }
    .approve-badge {
        background: #1b5e20; color: #a5d6a7;
        padding: 12px 24px; border-radius: 8px;
        font-size: 1.4rem; font-weight: bold; text-align: center;
    }
    .reject-badge {
        background: #b71c1c; color: #ef9a9a;
        padding: 12px 24px; border-radius: 8px;
        font-size: 1.4rem; font-weight: bold; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data Loading & Preprocessing ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("loan_approval_data.csv")

    # Impute
    num_cols  = df.select_dtypes(include="number").columns
    cat_cols  = df.select_dtypes(include="object").columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    df = df.drop("Applicant_ID", axis=1)

    # Encode target + Education_Level
    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    # One-hot encode categoricals
    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded    = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    # Feature Engineering
    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    return df, X, y, ohe, le

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }

    results = {}
    trained = {}
    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        y_pred = mdl.predict(X_test_s)
        results[name] = {
            "Accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred,    zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred,        zero_division=0), 4),
            "CM":        confusion_matrix(y_test, y_pred),
        }
        trained[name] = mdl

    return trained, results, scaler, X_train, X_test, y_train, y_test, X.columns.tolist()

# ─── Load everything ──────────────────────────────────────────────────────────
df_raw, X, y, ohe, le = load_and_preprocess()
trained_models, results, scaler, X_train, X_test, y_train, y_test, feature_cols = train_models(X, y)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 💳 CreditWise")
st.sidebar.markdown("*ML Loan Approval System*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Models", "Predict"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Records**\n\n# {len(df_raw):,}")
st.sidebar.markdown(f"**Features**\n\n# {X.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("📊 Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><p>Total Records</p><h2>{len(df_raw):,}</h2></div>', unsafe_allow_html=True)
    with col2:
        approved = int(df_raw["Loan_Approved"].sum())
        st.markdown(f'<div class="metric-card"><p>Approved Loans</p><h2>{approved:,}</h2></div>', unsafe_allow_html=True)
    with col3:
        rejected = len(df_raw) - approved
        st.markdown(f'<div class="metric-card"><p>Rejected Loans</p><h2>{rejected:,}</h2></div>', unsafe_allow_html=True)
    with col4:
        best_model = max(results, key=lambda m: results[m]["Precision"])
        best_prec  = results[best_model]["Precision"]
        st.markdown(f'<div class="metric-card"><p>Best Precision</p><h2>{best_prec:.0%}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df_raw.describe(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.title("🔍 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Loan Approval Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        df_raw["Loan_Approved"].value_counts().plot.pie(
            ax=ax, autopct="%1.1f%%", labels=["Rejected", "Approved"],
            colors=["#ef5350", "#66bb6a"], startangle=90
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Applicant Income Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df_raw["Applicant_Income"], bins=30, color="#4fc3f7", edgecolor="white")
        ax.set_xlabel("Income")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Credit Score vs Loan Approved")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=df_raw, x="Credit_Score", hue="Loan_Approved",
                     bins=20, multiple="dodge", ax=ax,
                     palette=["#ef5350", "#66bb6a"])
        st.pyplot(fig)

    with col4:
        st.subheader("Savings by Approval")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.boxplot(data=df_raw, x="Loan_Approved", y="Savings", ax=ax,
                    palette=["#ef5350", "#66bb6a"])
        ax.set_xticklabels(["Rejected", "Approved"])
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(14, 6))
    num_df = df_raw.select_dtypes(include="number")
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5)
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Models
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Models":
    st.title("🤖 Model Performance")

    metrics_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1 Score"]]
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#1b5e20"), use_container_width=True)

    st.markdown("---")

    st.subheader("Metric Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    metrics_df[["Accuracy", "Precision", "Recall", "F1 Score"]].plot(
        kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7
    )
    ax.set_xticklabels(metrics_df.index, rotation=0)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.set_title("Model Comparison")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for idx, (name, res) in enumerate(results.items()):
        with cols[idx]:
            st.markdown(f"**{name}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(res["CM"], annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Rejected", "Approved"],
                        yticklabels=["Rejected", "Approved"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict":
    st.title("🎯 Predict Loan")

    model_choice = st.selectbox("Model", list(trained_models.keys()))

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        applicant_income   = st.number_input("Applicant Income",    min_value=0, value=5000,   step=500)
        coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=0,      step=500)
        loan_amount        = st.number_input("Loan Amount",          min_value=0, value=150000, step=5000)
        loan_term          = st.number_input("Term (months)",        min_value=1, value=120,    step=12)
        dti_ratio          = st.slider("DTI Ratio",                  0.0, 1.0, 0.35, 0.01)

    with col2:
        age              = st.number_input("Age",           min_value=18, max_value=80, value=35)
        loan_purpose     = st.selectbox("Purpose",          ["Home", "Education", "Business", "Personal", "Other"])
        savings          = st.number_input("Savings",       min_value=0, value=10000, step=1000)
        collateral_value = st.number_input("Collateral Value", min_value=0, value=50000, step=5000)
        credit_score     = st.number_input("Credit Score",  min_value=300, max_value=900, value=650)

    with col3:
        dependents        = st.number_input("Dependents",       min_value=0, max_value=10, value=0)
        property_area     = st.selectbox("Property Area",       ["Urban", "Rural", "Semiurban"])
        employment_status = st.selectbox("Employment Status",   ["Employed", "Self-Employed", "Unemployed"])
        education_level   = st.selectbox("Education Level",     ["Graduate", "Not Graduate"])
        gender            = st.selectbox("Gender",              ["Male", "Female"])
        marital_status    = st.selectbox("Marital Status",      ["Married", "Single", "Divorced"])
        employer_category = st.selectbox("Employer Category",   ["Government", "Private", "NGO"])
        existing_loans    = st.number_input("Existing Loans",   min_value=0, max_value=10, value=0)

    if st.button("🔍 Predict", use_container_width=True):
        edu_encoded = 1 if education_level == "Graduate" else 0

        ohe_input = pd.DataFrame([{
            "Employment_Status":  employment_status,
            "Marital_Status":     marital_status,
            "Loan_Purpose":       loan_purpose,
            "Property_Area":      property_area,
            "Gender":             gender,
            "Employer_Category":  employer_category,
        }])
        ohe_cols_encoded = ohe.transform(ohe_input)
        ohe_df = pd.DataFrame(ohe_cols_encoded, columns=ohe.get_feature_names_out(
            ["Employment_Status", "Marital_Status", "Loan_Purpose",
             "Property_Area", "Gender", "Employer_Category"]
        ))

        dti_sq    = dti_ratio ** 2
        credit_sq = credit_score ** 2

        base = pd.DataFrame([{
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age":                age,
            "Dependents":         dependents,
            "Existing_Loans":     existing_loans,
            "Savings":            savings,
            "Collateral_Value":   collateral_value,
            "Loan_Amount":        loan_amount,
            "Loan_Term":          loan_term,
            "Education_Level":    edu_encoded,
            "DTI_Ratio_sq":       dti_sq,
            "Credit_Score_sq":    credit_sq,
        }])

        input_df = pd.concat([base.reset_index(drop=True),
                              ohe_df.reset_index(drop=True)], axis=1)

        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        input_scaled = scaler.transform(input_df)

        model = trained_models[model_choice]
        pred  = model.predict(input_scaled)[0]
        prob  = model.predict_proba(input_scaled)[0]

        st.markdown("---")
        st.subheader("Result")
        if pred == 1:
            st.markdown('<div class="approve-badge">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="reject-badge">❌ LOAN REJECTED</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Approval Probability",  f"{prob[1]*100:.1f}%")
        with c2:
            st.metric("Rejection Probability", f"{prob[0]*100:.1f}%")

        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh([""], [prob[1]], color="#66bb6a", label="Approved")
        ax.barh([""], [prob[0]], left=[prob[1]], color="#ef5350", label="Rejected")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.legend(loc="upper right")
        ax.set_title(f"Model: {model_choice}")
        st.pyplot(fig)
