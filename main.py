import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import re

# === 1. Case-Shiller Index ===
case_shiller = pd.read_csv("CSUSHPISA.csv")
case_shiller.columns = ["Date", "Case_Shiller_Index"]
case_shiller["Date"] = pd.to_datetime(case_shiller["Date"])
case_shiller.set_index("Date", inplace=True)
case_shiller["Case_Shiller_Change"] = case_shiller["Case_Shiller_Index"].pct_change()
print("Case-Shiller shape:", case_shiller.shape)

# === 2. Unemployment Rate ===
unemployment = pd.read_csv("unemployment_rate.csv")
unemployment["Date"] = pd.to_datetime(unemployment["Year"].astype(str) + "-" + unemployment["Period"].str.extract(r"M(\d+)")[0])
unemployment = unemployment[["Date", "Value"]].rename(columns={"Value": "Unemployment_Rate"})
unemployment.set_index("Date", inplace=True)
print("Unemployment shape:", unemployment.shape)

# === 3. Consumer Sentiment ===
try:
    sentiment = pd.read_csv("consumer_sentiment.csv", sep='\t', skiprows=1)
    print("Sentiment raw columns:", sentiment.columns.tolist())
    print("Sentiment raw sample:\n", sentiment.head())
    sentiment = sentiment.rename(columns={"Index": "Consumer_Sentiment"})
    sentiment["Date"] = pd.to_datetime(
        sentiment["Year"].astype(str) + "-" + sentiment["Month"].astype(str) + "-01",
        format="%Y-%m-%d"
    )
    sentiment = sentiment[["Date", "Consumer_Sentiment"]].set_index("Date")
except Exception as e:
    print(f"Error loading/parsing consumer_sentiment.csv: {e}")
    sentiment = pd.DataFrame()
print("Sentiment shape:", sentiment.shape)
print("Sentiment sample:\n", sentiment.head())

# === 4. Population ===
population = pd.read_csv("population.csv")
population = population[population["Unnamed: 1"] == "US"]
population = population.rename(columns={"Unnamed: 2": "Year", "Unnamed: 3": "Population"})
population["Date"] = pd.to_datetime(population["Year"].astype(str) + "-01-01")
population["Population"] = population["Population"].str.replace(",", "").astype(float)
population = population[["Date", "Population"]].set_index("Date").resample("MS").ffill()
print("Population shape:", population.shape)

# === 5. Housing Starts ===
df_housing = pd.read_excel("housing_starts.xlsx", skiprows=5)
df_housing = df_housing.rename(columns={"Unnamed: 0": "RawDate", "Total": "Housing_Starts"})
df_housing = df_housing[["RawDate", "Housing_Starts"]]
df_housing = df_housing[df_housing["RawDate"].notna()]

clean_data = []
current_year = None
for i, row in df_housing.iterrows():
    raw = str(row["RawDate"]).strip()
    year_match = re.match(r"^\d{4}", raw)
    if year_match:
        current_year = year_match.group(0)
        continue
    month_match = re.match(r"^(January|February|March|April|May|June|July|August|September|October|November|December)", raw)
    if month_match and current_year:
        clean_data.append({
            "Month": month_match.group(0),
            "Year": current_year,
            "Housing_Starts": row["Housing_Starts"]
        })
clean_df = pd.DataFrame(clean_data)
clean_df["Date"] = pd.to_datetime(clean_df["Month"] + " " + clean_df["Year"], format="%B %Y")
clean_df = clean_df[["Date", "Housing_Starts"]]
clean_df["Housing_Starts"] = pd.to_numeric(clean_df["Housing_Starts"], errors="coerce")
clean_df = clean_df.dropna()
clean_df.set_index("Date", inplace=True)
print("Housing Starts shape:", clean_df.shape)
print("Housing Starts sample:\n", clean_df.head())

# === 6. Interest Rates ===
weekly_data = pd.read_excel("historicalweeklydata.xlsx", skiprows=6)
weekly_data["Date"] = pd.to_datetime(weekly_data["Week"], errors="coerce")
weekly_data = weekly_data[["Date", "FRM"]].dropna()
weekly_data.set_index("Date", inplace=True)
monthly_interest = weekly_data.resample("MS").mean().rename(columns={"FRM": "Interest_Rate"})
print("Interest Rates shape:", monthly_interest.shape)

# === Check Date Ranges ===
print("\nDate Ranges:")
print("Case-Shiller:", case_shiller.index.min(), "to", case_shiller.index.max())
print("Unemployment:", unemployment.index.min(), "to", unemployment.index.max())
print("Sentiment:", sentiment.index.min(), "to", sentiment.index.max() if not sentiment.empty else "Empty")
print("Population:", population.index.min(), "to", population.index.max())
print("Housing Starts:", clean_df.index.min(), "to", clean_df.index.max())
print("Interest Rates:", monthly_interest.index.min(), "to", monthly_interest.index.max())

# === Merge All Datasets ===
df = case_shiller[["Case_Shiller_Change"]].join(
    [unemployment, sentiment, population, clean_df, monthly_interest], how="left"
)
print("\nMerged shape before dropna:", df.shape)
print("Missing values before dropna:\n", df.isna().sum())

# Drop rows with NaN in target variable and impute features
df = df.dropna(subset=["Case_Shiller_Change"])
df = df.ffill().bfill()
print("Shape after imputation:", df.shape)
print("Missing values after imputation:\n", df.isna().sum())

# === Modeling ===
available_features = [col for col in ["Unemployment_Rate", "Consumer_Sentiment", "Population", "Housing_Starts", "Interest_Rate"] if col in df.columns]
X = df[available_features]
y = df["Case_Shiller_Change"]
if df.shape[0] > 0:
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # === Evaluation ===
    print("\nR² Score:", r2_score(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

    # === Feature Importance ===
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)
    print("\nFeature Importances (Effect on Home Price % Change):")
    print(coef_df)

    # === Interpretation ===
    print("\nInterpretation of Coefficients:")
    for _, row in coef_df.iterrows():
        feature = row["Feature"]
        coef = row["Coefficient"]
        sign = "increases" if coef > 0 else "decreases"
        print(f"- A 1-unit increase in {feature} {sign} monthly home price % change by {abs(coef):.6f}.")

    # === Plot Predictions ===
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, y, label="Actual", linewidth=2)
    plt.plot(df.index, y_pred, label="Predicted", linestyle="--")
    plt.title("Predicted vs Actual Monthly Home Price % Change")
    plt.xlabel("Date")
    plt.ylabel("Monthly % Change")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # === Final Outputs ===
    print("\nFinal features and target shape:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("\nMissing values in each column:")
    print(df[available_features + ["Case_Shiller_Change"]].isna().sum())
    print("\nSample rows (after imputation):")
    print(df.head())
else:
    print("Error: Merged DataFrame is empty. Check date alignment or data availability.")
