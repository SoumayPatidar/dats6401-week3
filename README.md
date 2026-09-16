# DATS 6401: Week 3 Homework
## Tabular & Multivariate Data — Correlation Heatmap & PCA

**Author:** Soumay Patidar  
**Course:** Visualization of Complex Data (DATS 6401)  
**Dataset:** Formula 1 World Championship (1950–2024)

### What This App Does

Aggregates F1 race results into a driver-level summary with 10 numeric career statistics, then:

- Shows a **correlation heatmap** of all 10 variables
- Runs **PCA** (with standardization) to project drivers onto 2D
- Includes a **color-by widget** to explore groupings
- Provides a **scree plot** (bonus) and written interpretation

### Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
