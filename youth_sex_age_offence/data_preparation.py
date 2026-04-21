import pandas as pd

file_path = "./youth_offenders_dataset.xlsx"

all_data = []

# Helper function
def process_block(df, sex, start_row, end_row):
    block = df.iloc[start_row:end_row].copy()
    
    # Age categories
    age_categories = block.iloc[:, 0]

    # Number columns (B–R → index 1 to 17)
    number_df = block.iloc[:, 1:18]
    number_df.columns = years

    # Rate columns (S–AI → index 18 onward)
    rate_df = block.iloc[:, 18:18+len(years)]
    rate_df.columns = years

    # Convert to long format
    number_long = number_df.copy()
    number_long["Age Group"] = age_categories.values
    number_long = number_long.melt(id_vars="Age Group",
                                  var_name="Year",
                                  value_name="Number")

    rate_long = rate_df.copy()
    rate_long["Age Group"] = age_categories.values
    rate_long = rate_long.melt(id_vars="Age Group",
                              var_name="Year",
                              value_name="Rate")

    merged = pd.merge(number_long, rate_long,
                      on=["Age Group", "Year"])

    merged["Sex"] = sex
    return merged


# Read all sheets
xls = pd.ExcelFile(file_path)

df = pd.read_excel(file_path, sheet_name=xls.sheet_names[6], header=None)

# Extract year labels from row 6
year_row = df.iloc[5, 1:18]
years = [str(y) for y in year_row]

blocks = [
    ("Male", 7, 10),
    ("Female", 12, 15),
    ("Unknown", 17, 20)
]

# Process each block
for sex, start, end in blocks:
    data = process_block(df, sex, start, end)
    all_data.append(data)

# Combine all states
final_df = pd.concat(all_data, ignore_index=True)

# Drop rows where both Number and Rate are missing
final_df = final_df.dropna(subset=["Number", "Rate"], how="all")

# -----------------------------
# Convert Number and Rate to numeric
# -----------------------------

final_df["Number"] = final_df["Number"].astype(int)
final_df["Rate"] = pd.to_numeric(final_df["Rate"], errors="coerce")

# Extract starting year (e.g., 2008 from "2008–09")
final_df["Year"] = final_df["Year"].str[:4].astype(int)

final_df["Sex"] = final_df["Sex"].str.strip()

final_df.reset_index(drop=True, inplace=True)

# Save
print(f"Cleaned dataset shape: {final_df.shape}")

final_df.to_csv("preprocessed_data.csv", index=False)

print("Dataset cleaned successfully!")