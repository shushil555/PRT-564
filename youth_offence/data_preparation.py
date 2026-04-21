import pandas as pd

file_path = "./youth_offenders_dataset.xlsx"

# Define states by format
states_format1 = ["New South Wales", "Queensland", "South Australia", "Victoria"]
states_format2 = ["Tasmania", "Western Australia", "Northern Territory", "Australian Capital Territory"]

all_data = []

# Helper function
def process_block(df, state_name, start_row, end_row):
    block = df.iloc[start_row:end_row].copy()
    
    # Offence names
    offences = block.iloc[:, 0]

    # Number columns (B–R → index 1 to 17)
    number_df = block.iloc[:, 1:18]
    number_df.columns = years

    # Rate columns (S–AI → index 18 onward)
    rate_df = block.iloc[:, 18:18+len(years)]
    rate_df.columns = years

    # Convert to long format
    number_long = number_df.copy()
    number_long["Offence"] = offences.values
    number_long = number_long.melt(id_vars="Offence",
                                  var_name="Year",
                                  value_name="Number")

    rate_long = rate_df.copy()
    rate_long["Offence"] = offences.values
    rate_long = rate_long.melt(id_vars="Offence",
                              var_name="Year",
                              value_name="Rate")

    merged = pd.merge(number_long, rate_long,
                      on=["Offence", "Year"])

    merged["State"] = state_name

    return merged


# Read all sheets
xls = pd.ExcelFile(file_path)

df = pd.read_excel(file_path, sheet_name=xls.sheet_names[3], header=None)

# Extract year labels from row 6
year_row = df.iloc[5, 1:18]
years = [str(y) for y in year_row]

blocks = [
    ("New South Wales", 7, 23),
    ("Victoria", 27, 42),
    ("Queensland", 46, 61),
    ("South Australia", 65, 80),
    ("Western Australia", 84, 98),
    ("Tasmania", 102, 116),
    ("Northern Territory", 120, 134),
    ("Australian Capital Territory", 138, 152)
]

# Process each block
for state_name, start, end in blocks:
    data = process_block(df, state_name, start, end)
    all_data.append(data)

# Combine all states
final_df = pd.concat(all_data, ignore_index=True)

# Save
final_df.to_csv("clean_youth_data.csv", index=False)

print("Dataset created successfully!")