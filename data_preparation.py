import pandas as pd

file_path = "./dataset.xlsx"

# Define states by format
states_format1 = ["New South Wales", "Queensland", "South Australia", "Victoria"]
states_format2 = ["Tasmania", "Western Australia", "Northern Territory", "Australian Capital Territory"]

all_data = []

# Helper function
def process_block(df, state, sex, start_row, end_row):
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

    merged["State"] = state
    merged["Sex"] = sex

    return merged


# Read all sheets
xls = pd.ExcelFile(file_path)

for state in xls.sheet_names[2:10]:
    df = pd.read_excel(file_path, sheet_name=state, header=None)

    # Extract year labels from row 6
    year_row = df.iloc[5, 1:18]
    years = [str(y) for y in year_row]

    # Format 1 states
    if state in states_format1:
        blocks = [
            ("Male", 7, 23),
            ("Female", 25, 41),
            ("Unknown", 43, 59)
        ]

    # Format 2 states
    else:
        blocks = [
            ("Male", 7, 22),
            ("Female", 24, 39),
            ("Unknown", 41, 46)
        ]

    # Process each block
    for sex, start, end in blocks:
        data = process_block(df, state, sex, start, end)
        all_data.append(data)

# Combine all states
final_df = pd.concat(all_data, ignore_index=True)

# Save
final_df.to_csv("clean_crime_data.csv", index=False)

print("Dataset created successfully!")