import pandas as pd

file_path = "./population.xlsx"

# Load excel file
xls = pd.ExcelFile(file_path)
df = pd.read_excel(file_path, sheet_name="Data1", header=None, skiprows=10)

# State, male population, female population
state_blocks = [
    ("New South Wales",              1, 10),
    ("Victoria",                     2, 11),
    ("Queensland",                   3, 12),
    ("South Australia",              4, 13),
    ("Western Australia",            5, 14),
    ("Tasmania",                     6, 15),
    ("Northern Territory",           7, 16),
    ("Australian Capital Territory", 8, 17),
]

# Filter to only include June data from 2008-2024 and extract years and row indices
df[0] = pd.to_datetime(df[0], errors="coerce")
mask  = (df[0].dt.month == 6) & df[0].dt.year.between(2008, 2024)
dates = df.loc[mask, 0].dt.year.values          # array of years
idx   = df.index[mask]                           # actual row indices

# Extract population data for each state and reshape to long format
frames = []

for state, male_col, female_col in state_blocks:
    male_pop   = df.loc[idx+4, male_col].values
    female_pop = df.loc[idx+4, female_col].values

    state_df = pd.DataFrame({
        "State":      state,
        "Year":       dates,
        "Sex":        "Male",
        "Population": male_pop,
    })
    state_df = pd.concat([state_df, pd.DataFrame({
        "State":      state,
        "Year":       dates,
        "Sex":        "Female",
        "Population": female_pop,
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
print(final_df.tail(20).to_string(index=False))

final_df.to_csv("population_long_format.csv", index=False)
print("\nData saved successfully.")