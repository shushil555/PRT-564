import pandas as pd

file_path = "./population.xlsx"

# Read excel file
xls = pd.ExcelFile(file_path)

df = pd.read_excel(file_path, sheet_name='Data1', header=None, skiprows=10)


# Define states by location of male and female population columns
state_blocks = [
    ("New South Wales", 1, 10),
    ("Victoria", 2, 11),
    ("Queensland", 3, 12),
    ("South Australia", 4, 13),
    ("Western Australia", 5, 14),
    ("Tasmania", 6, 15),
    ("Northern Territory", 7, 16),
    ("Australian Capital Territory", 8, 17)
]

# Filter entire dataset to only include June data from 2008-2024
date_mask = (df[0].dt.year >= 2008) & (df[0].dt.month == 6)
date_filtered_df = df[date_mask]

final_data = list()

# For each state, extract population data and reshape to long format
for state, male, female in state_blocks:
    for i in range(len(date_filtered_df)):
        if date_filtered_df.iloc[i, 0].year < 2025:
            final_data.append({
                "State": state,
                "Year": date_filtered_df.iloc[i, 0].year,
                "Sex": "Male",
                "Population": date_filtered_df.iloc[i+1, male]
            })
            final_data.append({
                "State": state,
                "Year": date_filtered_df.iloc[i, 0].year,
                "Sex": "Female",
                "Population": date_filtered_df.iloc[i+1, female]
            })

final_df = pd.DataFrame(final_data, columns=["State", "Year", "Sex", "Population"])
print(final_df.tail(20))

final_df.to_csv("population_long_format.csv", index=False)
print("\nData saved successfully.")