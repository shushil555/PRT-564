import pandas as pd

file_path = "./unemployment.xlsx"

# Load excel file
xls = pd.ExcelFile(file_path)
df = pd.read_excel(file_path, sheet_name="Data2", header=None, skiprows=10)

# State, male unemployment rate, female unemployment rate
state_blocks = [
    ("New South Wales",              5, 30),
    ("Victoria",                     8, 33),
    ("Queensland",                   11, 36),
    ("South Australia",              14, 39),
    ("Western Australia",            17, 42),
    ("Tasmania",                     20, 45),
    ("Northern Territory",           22, 47),
    ("Australian Capital Territory", 24, 49)
]

# Filter to only include June data from 2008-2024 and extract years and row indices
df[0] = pd.to_datetime(df[0], errors="coerce")
mask  = (df[0].dt.month == 6) & df[0].dt.year.between(2008, 2024)
dates = df.loc[mask, 0].dt.year.values          # array of years
idx   = df.index[mask]                           # actual row indices

# print(df.loc[:, 175:224].head(10))  # Check the structure of the data around the unemployment rates

# Extract unemployment data for each state and reshape to long format
frames = []

data_start_index = 175  # Starting column index for unemployment rates

for state, male_col, female_col in state_blocks:
    male_unemployment_rate   = df.loc[idx+12, data_start_index+male_col].values
    female_unemployment_rate = df.loc[idx+12, data_start_index+female_col].values

    state_df = pd.DataFrame({
        "State":      state,
        "Year":       dates,
        "Sex":        "Male",
        "Unemployment_Rate": male_unemployment_rate,
    })
    state_df = pd.concat([state_df, pd.DataFrame({
        "State":      state,
        "Year":       dates,
        "Sex":        "Female",
        "Unemployment_Rate": female_unemployment_rate,
    })], ignore_index=True)

    frames.append(state_df)

# Combine all states into one DataFrame
final_df = (
    pd.concat(frames, ignore_index=True)
    .reset_index(drop=True)
)

print("Shape:", final_df.shape)
print("Years:", sorted(final_df["Year"].unique()))
print("States:", final_df["State"].nunique())
print()
print("First 5 rows of the unemployment data:")
print(final_df.head(5).to_string(index=False))

final_df.to_csv("unemployment_long_format.csv", index=False)
print("\nData saved successfully.")