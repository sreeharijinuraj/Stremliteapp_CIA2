import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Shopper Intent | Purchase Prediction",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path(__file__).parent / "models" / "online_shoppers_purchase_model.pkl"
FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def load_uploaded_data(upload):
    if upload is None:
        return None
    try:
        return pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None

def page_business():
    st.title("🛍️ Online Shopping Purchase Prediction")
    st.caption("A machine-learning application for understanding and predicting e-commerce purchase intent.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Analytics task", "Classification")
    c2.metric("Target", "Revenue")
    c3.metric("Dataset", "UCI Online Shoppers")
    st.subheader("Business problem")
    st.write(
        "E-commerce websites attract many visitors, but only a portion complete a purchase. "
        "The business challenge is to identify sessions that are more likely to convert, using "
        "browsing behaviour and session characteristics."
    )
    st.subheader("Analytics objective")
    st.write(
        "Use session-level behavioural features to estimate whether a website visit will result "
        "in a purchase (Revenue = True) or no purchase (Revenue = False). The prediction can "
        "support better-timed engagement, experience improvements, and campaign prioritization."
    )
    st.info(
        "This model estimates purchase intention from the dataset's session features. "
        "It should support—not replace—business judgment, and it does not prove that an intervention causes a purchase."
    )
    st.subheader("Dataset overview")
    st.markdown(
        "- **Source:** UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset\n"
        "- **Unit of analysis:** A website browsing session\n"
        "- **Target:** `Revenue` (purchase / no purchase)\n"
        "- **Model:** Saved scikit-learn preprocessing + Logistic Regression pipeline\n"
    )
    st.markdown("[Open the UCI dataset page](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)")
    st.subheader("Project team")
    names = ["Sreehari J", "Rhea Sibi", "Richin", "Diya", "Vandita"]
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        with col:
            st.markdown(f"<div style='padding:14px;border:1px solid #d9e2f2;border-radius:12px;text-align:center;background:#f6f8fc;color:#172554;font-weight:600'>{name}</div>", unsafe_allow_html=True)
    st.caption("Team members • Online Shopping Purchase Prediction Using Machine Learning")

def page_insights():
    st.title("📊 Data Insights")
    st.write(
        "Upload the team's cleaned dataset (CSV) to view data-driven summaries and visualizations. "
        "Charts are generated from the uploaded file; no results are fabricated when data is not supplied."
    )
    upload = st.file_uploader("Upload cleaned dataset CSV", type=["csv"], key="insights_csv")
    df = load_uploaded_data(upload)
    if df is None:
        st.info("No CSV uploaded yet. Use the cleaned dataset prepared by Member 2 / the common UCI dataset.")
        st.markdown("[Get the dataset from UCI](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)")
        return
    if "Revenue" not in df.columns:
        st.error("The uploaded CSV must contain a `Revenue` column.")
        return
    # Normalize common textual labels to boolean-like labels for charting.
    rev = df["Revenue"].astype(str).str.lower().map({
        "true": "Purchase", "1": "Purchase", "yes": "Purchase",
        "false": "No Purchase", "0": "No Purchase", "no": "No Purchase"
    }).fillna(df["Revenue"].astype(str))
    df = df.copy()
    df["Purchase outcome"] = rev
    total = len(df)
    purchase_rate = (rev == "Purchase").mean() * 100 if total else 0
    a, b, c = st.columns(3)
    a.metric("Sessions in uploaded data", f"{total:,}")
    b.metric("Purchase sessions", f"{(rev == 'Purchase').sum():,}")
    c.metric("Purchase rate", f"{purchase_rate:.2f}%")
    left, right = st.columns(2)
    with left:
        counts = df["Purchase outcome"].value_counts().rename_axis("Outcome").reset_index(name="Sessions")
        st.plotly_chart(px.bar(counts, x="Outcome", y="Sessions", title="Purchase outcome distribution"), use_container_width=True)
    with right:
        if "VisitorType" in df.columns:
            byvisitor = df.groupby("VisitorType", dropna=False)["Purchase outcome"].apply(lambda x: (x == "Purchase").mean() * 100).reset_index(name="Purchase rate (%)")
            st.plotly_chart(px.bar(byvisitor, x="VisitorType", y="Purchase rate (%)", title="Purchase rate by visitor type"), use_container_width=True)
        elif "Month" in df.columns:
            bymonth = df.groupby("Month", dropna=False)["Purchase outcome"].apply(lambda x: (x == "Purchase").mean() * 100).reset_index(name="Purchase rate (%)")
            st.plotly_chart(px.bar(bymonth, x="Month", y="Purchase rate (%)", title="Purchase rate by month"), use_container_width=True)
        else:
            st.warning("Add `VisitorType` or `Month` to the CSV to show a segment chart.")
    if "PageValues" in df.columns:
        st.plotly_chart(px.box(df, x="Purchase outcome", y="PageValues", title="PageValues by purchase outcome"), use_container_width=True)
    st.caption("Interpret these patterns as associations in session data, not causal effects.")

def page_prediction():
    st.title("🎯 Purchase Prediction")
    st.write("Enter a visitor session's observed activity. Values should represent one session.")
    model = load_model()
    with st.form("prediction_form"):
        st.subheader("Browsing activity")
        c1, c2, c3 = st.columns(3)
        with c1:
            administrative = st.number_input("Administrative pages", min_value=0, value=0, step=1)
            admin_duration = st.number_input("Administrative duration (seconds)", min_value=0.0, value=0.0)
            informational = st.number_input("Informational pages", min_value=0, value=0, step=1)
            info_duration = st.number_input("Informational duration (seconds)", min_value=0.0, value=0.0)
        with c2:
            product_related = st.number_input("Product-related pages", min_value=0, value=1, step=1)
            product_duration = st.number_input("Product-related duration (seconds)", min_value=0.0, value=0.0)
            bounce = st.number_input("Bounce rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
            exit_rate = st.number_input("Exit rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01, format="%.4f")
        with c3:
            page_values = st.number_input("PageValues", min_value=0.0, value=0.0, step=0.1)
            special_day = st.number_input("SpecialDay (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
            month = st.selectbox("Month", MONTHS)
            visitor_type = st.selectbox("Visitor type", VISITOR_TYPES, index=0)
        st.subheader("Session context")
        c4, c5, c6, c7 = st.columns(4)
        with c4:
            operating_system = st.selectbox("Operating system code", list(range(1, 9)), index=0)
        with c5:
            browser = st.selectbox("Browser code", list(range(1, 14)), index=0)
        with c6:
            region = st.selectbox("Region code", list(range(1, 10)), index=0)
        with c7:
            traffic_type = st.selectbox("Traffic type code", list(range(1, 21)), index=0)
        weekend = st.selectbox("Weekend session?", ["False", "True"])
        submitted = st.form_submit_button("Predict purchase intention", type="primary", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": admin_duration,
            "Informational": informational,
            "Informational_Duration": info_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration,
            "BounceRates": bounce,
            "ExitRates": exit_rate,
            "PageValues": page_values,
            "SpecialDay": special_day,
            "Month": month,
            "OperatingSystems": operating_system,
            "Browser": browser,
            "Region": region,
            "TrafficType": traffic_type,
            "VisitorType": visitor_type,
            "Weekend": weekend == "True",
        }])[FEATURES]
        try:
            prediction = model.predict(input_df)[0]
            # Display probability for the positive class only if model supports it.
            probability = None
            if hasattr(model, "predict_proba"):
                classes = list(model.classes_)
                positive_index = next((i for i, cls in enumerate(classes) if str(cls).lower() in ["true", "1", "yes"]), None)
                if positive_index is not None:
                    probability = float(model.predict_proba(input_df)[0][positive_index])
            is_purchase = str(prediction).lower() in ["true", "1", "yes"]
            if is_purchase:
                st.success("Prediction: Purchase Likely")
                st.write("This session is classified by the model as likely to result in a purchase.")
            else:
                st.warning("Prediction: Purchase Unlikely")
                st.write("This session is classified by the model as unlikely to result in a purchase.")
            if probability is not None:
                st.metric("Estimated purchase probability", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            st.caption("Prediction is based on the trained model and entered session data; it is not a guarantee of customer behaviour.")
        except Exception as exc:
            st.error(f"Prediction failed. Check model compatibility and feature inputs. Details: {exc}")

def main():
    st.sidebar.title("Shopper Intent")
    st.sidebar.caption("Online Shopping Purchase Prediction")
    page = st.sidebar.radio("Navigate", ["Business Problem", "Data Insights", "Prediction"])
    st.sidebar.divider()
    st.sidebar.markdown("**Project team**")
    st.sidebar.write("Sreehari J · Rhea Sibi · Richin · Diya · Chotu")
    if page == "Business Problem":
        page_business()
    elif page == "Data Insights":
        page_insights()
    else:
        page_prediction()

if __name__ == "__main__":
    main()
