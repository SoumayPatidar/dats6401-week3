# DATS 6401: Visualization of Complex Data
# Week 3 Homework: Tabular & Multivariate Data
# Author: Soumay Patidar
# Dataset: Formula 1 World Championship (1950–2024)

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.set_page_config(page_title="F1 Multivariate Analysis", page_icon="🏎️", layout="wide")

st.title("🏎️ F1 Driver Careers: Multivariate Analysis & PCA")
st.markdown(
    """
    ### About This Analysis

    This app explores **Formula 1 driver career statistics** as a multivariate
    dataset.  I aggregated the raw race results (1950–2024) into one row per
    driver, creating **10 numeric variables** that describe each driver's career:
    total races, wins, podiums, points, average grid/finish position, laps,
    DNFs, win rate, and podium rate.

    Only drivers with **30 or more career races** are included so that the
    statistics are meaningful (219 drivers total).

    The goal is to find **structure** in this high-dimensional data:
    - Which variables are correlated?
    - Can we project 10 dimensions down to 2 and still see meaningful groupings?
    """
)
@st.cache_data
def load_data():
    results = pd.read_csv("data/results.csv")
    races = pd.read_csv("data/races.csv")
    drivers = pd.read_csv("data/drivers.csv")
    status = pd.read_csv("data/status.csv")

    # merge tables
    df = (
        results
        .merge(races[["raceId", "year"]], on="raceId")
        .merge(drivers[["driverId", "forename", "surname", "nationality"]], on="driverId")
        .merge(status[["statusId", "status"]], on="statusId")
    )

    # clean up numeric columns
    df["driver"] = df["forename"] + " " + df["surname"]
    df["position_num"] = pd.to_numeric(df["position"], errors="coerce")
    df["grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["laps"] = pd.to_numeric(df["laps"], errors="coerce")

    # aggregate to one row per driver
    stats = df.groupby(["driver", "nationality"]).agg(
        total_races=("raceId", "nunique"),
        total_wins=("position_num", lambda x: (x == 1).sum()),
        total_podiums=("position_num", lambda x: (x <= 3).sum()),
        total_points=("points", "sum"),
        avg_grid=("grid", "mean"),
        avg_finish=("position_num", "mean"),
        total_laps=("laps", "sum"),
        dnf_count=("status", lambda x: (x != "Finished").sum()),
    ).reset_index()

    # only keep drivers with 30+ races
    stats = stats[stats["total_races"] >= 30].reset_index(drop=True)

    # add rate columns
    stats["win_rate"] = round(stats["total_wins"] / stats["total_races"] * 100, 2)
    stats["podium_rate"] = round(stats["total_podiums"] / stats["total_races"] * 100, 2)

    # group small nationalities as "Other" for cleaner plots
    top_nats = stats["nationality"].value_counts().head(8).index
    stats["nat_group"] = stats["nationality"].apply(
        lambda x: x if x in top_nats else "Other"
    )

    return stats


driver_stats = load_data()

# list of numeric columns we'll use
numeric_cols = [
    "total_races", "total_wins", "total_podiums", "total_points",
    "avg_grid", "avg_finish", "total_laps", "dnf_count",
    "win_rate", "podium_rate",
]
st.markdown("### Data Preview")
st.markdown(f"**{len(driver_stats)} drivers** with **{len(numeric_cols)} numeric variables** each.")
st.dataframe(driver_stats, use_container_width=True, height=250)

st.markdown("---")
# CORRELATION HEATMAP
st.markdown("### Correlation Heatmap")

fig1, ax1 = plt.subplots(figsize=(10, 8))
corr = driver_stats[numeric_cols].corr()
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    square=True,
    cbar_kws={"shrink": 0.7},
    ax=ax1,
)
ax1.set_title("Correlation Among Driver Career Statistics", fontsize=14)
plt.tight_layout()
st.pyplot(fig1)

st.markdown(
    """
    **What I see:**
    - **total_wins, total_podiums, total_points, win_rate, and podium_rate** are
      all strongly positively correlated (0.7–0.9+).  This makes sense, drivers
      who win a lot also get more podiums and more points.
    - **avg_grid and avg_finish** are positively correlated (0.78), drivers who
      qualify higher tend to finish higher.
    - **total_races and total_laps** are almost perfectly correlated (0.98),
      more races obviously means more laps driven.
    - **dnf_count** correlates with **total_races** (0.73), drivers who raced
      more also had more retirements, which is just a function of career length.
    - **avg_grid** has a *negative* correlation with win_rate, lower grid
      numbers (i.e., qualifying closer to the front) go with higher win rates.
    """
)

st.markdown("---")

# PCA PROJECTION
st.markdown("###PCA Projection to 2D")

st.markdown(
    """
    PCA (Principal Component Analysis) takes all 10 numeric variables and
    finds the two directions that capture the most variance.  This lets us
    plot 219 drivers on a 2D scatter and see if any natural groups emerge.

    **Important:** I standardized the data first (mean=0, std=1) so that
    variables on different scales (e.g., total_laps vs. win_rate) contribute
    equally.
    """
)
color_options = ["nat_group", "nationality"]
color_choice = st.selectbox(
    "Color the PCA scatter by:",
    color_options,
    index=0,
    format_func=lambda x: {
        "nat_group": "Nationality (grouped)",
        "nationality": "Nationality (all)",
    }[x],
)
X = driver_stats[numeric_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)

# build a dataframe for plotting
pca_df = pd.DataFrame({
    "PC1": pcs[:, 0],
    "PC2": pcs[:, 1],
    "driver": driver_stats["driver"],
    "color": driver_stats[color_choice],
})

# plot
fig2, ax2 = plt.subplots(figsize=(10, 7))
groups = pca_df["color"].unique()
for g in sorted(groups):
    mask = pca_df["color"] == g
    ax2.scatter(
        pca_df.loc[mask, "PC1"],
        pca_df.loc[mask, "PC2"],
        label=g,
        alpha=0.6,
        s=50,
    )

# label a few famous drivers so the plot is more interesting
famous = ["Lewis Hamilton", "Michael Schumacher", "Max Verstappen",
          "Ayrton Senna", "Fernando Alonso", "Sebastian Vettel"]
for _, row in pca_df[pca_df["driver"].isin(famous)].iterrows():
    ax2.annotate(
        row["driver"].split()[-1],  # just last name
        (row["PC1"], row["PC2"]),
        fontsize=8,
        ha="left",
        va="bottom",
    )

var1 = pca.explained_variance_ratio_[0]
var2 = pca.explained_variance_ratio_[1]
ax2.set_xlabel(f"PC1 ({var1:.1%} variance)", fontsize=12)
ax2.set_ylabel(f"PC2 ({var2:.1%} variance)", fontsize=12)
ax2.set_title("F1 Drivers Projected onto First Two Principal Components", fontsize=14)
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
plt.tight_layout()
st.pyplot(fig2)

st.markdown(f"**Explained variance:** PC1 = {var1:.1%}, PC2 = {var2:.1%}, "
            f"together = {var1 + var2:.1%}")
# SCREE PLOT (bonus)
st.markdown("---")
st.markdown("### Scree Plot (Bonus)")

pca_full = PCA().fit(X_scaled)
fig3, ax3 = plt.subplots(figsize=(8, 4))
components = range(1, len(pca_full.explained_variance_ratio_) + 1)
ax3.bar(components, pca_full.explained_variance_ratio_, color="steelblue", alpha=0.7)
ax3.plot(components, np.cumsum(pca_full.explained_variance_ratio_),
         "ro-", markersize=6, label="Cumulative")
ax3.set_xlabel("Principal Component")
ax3.set_ylabel("Explained Variance Ratio")
ax3.set_title("Scree Plot — How Much Variance Each Component Captures")
ax3.legend()
ax3.set_xticks(list(components))
plt.tight_layout()
st.pyplot(fig3)
# INTERPRETATION
st.markdown("---")
st.markdown("### Interpretation")

st.markdown(
    f"""
    **What does PC1 represent?**

    PC1 captures **{var1:.1%}** of the total variance and seems to represent
    **career success and volume**.  Drivers on the far right of the scatter
    (Hamilton, Schumacher, Verstappen, Alonso) have high values across wins,
    podiums, points, and total races.  Drivers on the left had shorter or
    less successful careers.  PC1 basically separates "legends" from
    "mid-field" and "short-career" drivers.

    **What does PC2 represent?**

    PC2 captures **{var2:.1%}** of the variance and seems to pick up a
    contrast between **career length vs. win efficiency**.  Some drivers
    raced for many years but had low win/podium rates (high total_races,
    low win_rate), pulling them in one direction on PC2.  Others had short
    but dominant careers (fewer races but high win_rate), pulling the other
    way.

    **Overall structure:**

    The PCA scatter doesn't cluster strongly by nationality — driver
    performance depends more on the car and the era than on where the driver
    is from.  The main structure is a "success gradient" from left to right,
    with the all-time greats clearly separated from the rest of the field.
    This is something no single variable shows on its own, but PCA combines
    all 10 variables to reveal it cleanly.
    """
)
st.markdown("---")
st.caption(
    "DATS 6401- Week 3 Homework - Soumay Patidar "
    "Data: Ergast F1 Database (1950–2024)"
)
