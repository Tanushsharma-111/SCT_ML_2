
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Mall Customer Analysis", page_icon="🏪", layout="wide")

st.markdown("""
<style>
h1, h2, h3 { color: #B71C1C; }
.stButton > button { background-color: #B71C1C; color: white; border-radius: 8px; border: none; padding: 0.4rem 1.2rem; }
.stButton > button:hover { background-color: #7F0000; }
</style>
""", unsafe_allow_html=True)

# loading the dataset
buyer_data = pd.read_csv("Mall_Customers.csv")

# loading trained model and scaler
seg_model = pickle.load(open("model.pkl", "rb"))
feat_scaler = pickle.load(open("scaler.pkl", "rb"))

# sidebar setup
st.sidebar.title("Mall Customer Analysis")
st.sidebar.markdown("---")
nav_choice = st.sidebar.radio("Navigate", ["Home", "Dataset", "Insights", "Elbow Method", "Clusters", "Predict"])
st.sidebar.markdown("---")
st.sidebar.markdown("Algorithm: K-Means Clustering  \nDataset: Mall Customers")

# home page
if nav_choice == "Home":
    st.title("Mall Customer Segmentation")
    st.markdown("This app segments customers based on their Annual Income and Spending Score using K-Means Clustering.")
    st.markdown("---")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customers", len(buyer_data))
    m2.metric("Segments", 5)
    m3.metric("Algorithm", "K-Means")

    st.markdown("---")
    st.image("https://images.unsplash.com/photo-1607082348824-0a96f2a4b9da?w=1000", use_container_width=True)

# dataset page
elif nav_choice == "Dataset":
    st.title("Customer Records")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        gender_filter = st.selectbox("Filter by Gender", ["All", "Male", "Female"])
    with col_b:
        low_inc = int(buyer_data["Annual Income (k$)"].min())
        high_inc = int(buyer_data["Annual Income (k$)"].max())
        inc_band = st.slider("Income Band (k$)", low_inc, high_inc, (low_inc, high_inc))

    # filtering based on user selection
    view_data = buyer_data.copy()
    if gender_filter != "All":
        view_data = view_data[view_data["Gender"] == gender_filter]
    view_data = view_data[
        (view_data["Annual Income (k$)"] >= inc_band[0]) &
        (view_data["Annual Income (k$)"] <= inc_band[1])
    ]

    st.write(f"Showing **{len(view_data)}** of **{len(buyer_data)}** customers")
    st.dataframe(view_data, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=view_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_customers.csv",
        mime="text/csv"
    )

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Dimensions")
        st.write(view_data.shape)
    with col_d:
        st.subheader("Null Values")
        st.write(view_data.isnull().sum())

# insights page
elif nav_choice == "Insights":
    st.title("Exploratory Analysis")

    tab_inc, tab_score, tab_corr = st.tabs(["Annual Income", "Spending Score", "Correlation"])

    with tab_inc:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(buyer_data["Annual Income (k$)"], kde=True, color="#B71C1C", ax=ax)
        ax.set_title("Annual Income Distribution (k$)")
        st.pyplot(fig)

    with tab_score:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(buyer_data["Spending Score (1-100)"], kde=True, color="#E65100", ax=ax)
        ax.set_title("Spending Score Distribution")
        st.pyplot(fig)

    with tab_corr:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(buyer_data.corr(numeric_only=True), annot=True, cmap="OrRd", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)

# elbow method page
elif nav_choice == "Elbow Method":
    st.title("Elbow Method")
    st.markdown("Used to find the best number of clusters for this data.")

    feature_matrix = buyer_data[["Annual Income (k$)", "Spending Score (1-100)"]]
    norm_scaler = StandardScaler()
    scaled_features = norm_scaler.fit_transform(feature_matrix)

    # running kmeans for k=1 to 10 and storing inertia
    inertia_vals = []
    k_range = range(1, 11)
    for k_val in k_range:
        km_temp = KMeans(n_clusters=k_val, init="k-means++", n_init=10, random_state=42)
        km_temp.fit(scaled_features)
        inertia_vals.append(km_temp.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), inertia_vals, marker="o", color="#B71C1C", linewidth=2)
    ax.axvline(x=5, color="#E65100", linestyle="--", label="Best K = 5")
    ax.set_title("Elbow Curve")
    ax.set_xlabel("Number of Segments (K)")
    ax.set_ylabel("Within-Cluster Sum of Squares")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.info("K = 5 gives the best result based on the elbow curve above.")

# segment data page
elif nav_choice == "Clusters":
    st.title("Customer Segments")

    feat_cols = buyer_data[["Annual Income (k$)", "Spending Score (1-100)"]]
    segment_labels = seg_model.predict(feat_scaler.transform(feat_cols))
    buyer_data["Segment"] = segment_labels

    PALETTE = ["#B71C1C", "#E65100", "#1565C0", "#4A148C", "#1B5E20"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for seg_id in range(5):
        idx = segment_labels == seg_id
        ax.scatter(feat_cols.values[idx, 0], feat_cols.values[idx, 1],
                   c=PALETTE[seg_id], s=80, label=f"Segment {seg_id}", alpha=0.85)

    # converting centroids back to original scale
    centroids_orig = feat_scaler.inverse_transform(seg_model.cluster_centers_)
    ax.scatter(centroids_orig[:, 0], centroids_orig[:, 1], s=250, c="black",
               marker="X", label="Centroids", zorder=5)

    ax.set_title("Mall Customer Segments")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend()
    ax.grid(alpha=0.2)
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Segment Summary")

    SEG_NAMES = {
        0: "Elite Spender",
        1: "Economy Shopper",
        2: "Regular Buyer",
        3: "Untapped Potential",
        4: "Everyday Consumer"
    }
    seg_summary = buyer_data.groupby("Segment")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(1)
    seg_summary["Count"] = buyer_data["Segment"].value_counts().sort_index()
    seg_summary["Category"] = seg_summary.index.map(SEG_NAMES)
    st.dataframe(seg_summary[["Category", "Annual Income (k$)", "Spending Score (1-100)", "Count"]], use_container_width=True)

# predict page
elif nav_choice == "Predict":
    st.title("Predict Customer Segment")
    st.markdown("Adjust the sliders and click predict.")

    col_x, col_y = st.columns(2)
    with col_x:
        annual_income = st.slider("Annual Income (k$)", 10, 150, 60)
    with col_y:
        spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

    SEGMENT_DESCRIPTIONS = {
        0: ("Elite Spender",      "High income and high spending — most valuable customers."),
        1: ("Economy Shopper",    "Low income and low spending — very price-conscious buyers."),
        2: ("Regular Buyer",      "Average income and spending — the typical customer."),
        3: ("Untapped Potential", "High income but low spending — target with premium offers."),
        4: ("Impulsive Buyer",    "Low income but high spending — driven by impulse purchases."),
    }

    if st.button("Predict"):
        # predicting segment based on slider inputs
        input_df = pd.DataFrame([[annual_income, spending_score]],
                                columns=["Annual Income (k$)", "Spending Score (1-100)"])
        input_vec = feat_scaler.transform(input_df)
        predicted_seg = int(seg_model.predict(input_vec)[0])
        seg_name, seg_desc = SEGMENT_DESCRIPTIONS.get(predicted_seg, (f"Segment {predicted_seg}", ""))
        st.success(f"Segment {predicted_seg} — {seg_name}")
        st.info(seg_desc)

        # showing where this customer falls on the map
        all_feats = buyer_data[["Annual Income (k$)", "Spending Score (1-100)"]]
        all_seg_labels = seg_model.predict(feat_scaler.transform(all_feats))
        PALETTE = ["#B71C1C", "#E65100", "#1565C0", "#4A148C", "#1B5E20"]

        fig, ax = plt.subplots(figsize=(7, 4))
        for seg_id in range(5):
            idx = all_seg_labels == seg_id
            ax.scatter(all_feats.values[idx, 0], all_feats.values[idx, 1],
                       c=PALETTE[seg_id], s=40, alpha=0.4)
        ax.scatter(annual_income, spending_score, s=300, c="yellow", marker="*",
                   zorder=6, label="This Customer", edgecolors="black")
        ax.set_xlabel("Annual Income (k$)")
        ax.set_ylabel("Spending Score (1-100)")
        ax.set_title("Where does this customer fall?")
        ax.legend()
        ax.grid(alpha=0.2)
        st.pyplot(fig)