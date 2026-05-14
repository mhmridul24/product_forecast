import pandas as pd

# =========================
# READ EXCEL FILE
# =========================

file_path = "data/Forecasting_2026.xlsx"

usage_df = pd.read_excel(
    file_path,
    sheet_name="Usage",
    header=None
)

# =========================
# REMOVE EMPTY ROWS/COLUMNS
# =========================

usage_df = usage_df.dropna(how='all')

usage_df = usage_df.dropna(axis=1, how='all')

# Reset index
usage_df = usage_df.reset_index(drop=True)

print("\n===== CLEANED DATA =====")
print(usage_df.head(10))

# =========================
# EXTRACT DATES
# =========================

dates = usage_df.iloc[0, 2:].values

print("\n===== DATES =====")
print(dates[:10])

# =========================
# EXTRACT PRODUCTS
# =========================

products = usage_df.iloc[2:, 1].values

print("\n===== PRODUCTS =====")
print(products)

# =========================
# EXTRACT USAGE VALUES
# =========================

usage_values = usage_df.iloc[2:, 2:]

# =========================
# CREATE CLEAN DATASET
# =========================

clean_data = []

for product_index, product in enumerate(products):

    for date_index, date in enumerate(dates):

        usage = usage_values.iloc[product_index, date_index]

        # Skip empty values
        if pd.isna(usage):
            continue

        clean_data.append({
            "Date": pd.to_datetime(date),
            "Product": product,
            "Usage": usage
        })

# =========================
# CREATE DATAFRAME
# =========================

clean_df = pd.DataFrame(clean_data)

# =========================
# SORT DATA
# =========================

clean_df = clean_df.sort_values(by=["Product", "Date"])

# =========================
# ADD EXTRA FEATURES
# =========================

clean_df["DayName"] = clean_df["Date"].dt.day_name()

clean_df["Month"] = clean_df["Date"].dt.month

clean_df["Year"] = clean_df["Date"].dt.year

# =========================
# SAVE CLEAN DATASET
# =========================

clean_df.to_csv("clean_usage_data.csv", index=False)

print("\n===== FINAL CLEAN DATA =====")
print(clean_df.head())

print("\nDataset cleaned successfully!")
print("\nSaved as: clean_usage_data.csv")