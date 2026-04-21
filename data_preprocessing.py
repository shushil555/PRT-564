import pandas as pd
import numpy as np
import re

# Load dataset
df = pd.read_csv("clean_crime_data.csv")


# 1. Clean Offence Names


def clean_offence(name):
    if pd.isna(name):
        return name
    
    name = str(name)
    
    # Remove leading numbers (e.g., "01 ")
    name = re.sub(r'^\d+\s+', '', name)
    
    # Remove footnotes like (e), (f), etc.
    name = re.sub(r'\(.*?\)', '', name)
    
    # Remove extra spaces
    name = name.strip()
    
    return name

df["Offence"] = df["Offence"].apply(clean_offence)


# 2. Handle 'np' values


df.replace("np", np.nan, inplace=True)


# 3. Convert Number and Rate to numeric


df["Number"] = pd.to_numeric(df["Number"], errors="coerce")
df["Rate"] = pd.to_numeric(df["Rate"], errors="coerce")


# 4. Clean Year column


# Extract starting year (e.g., 2008 from "2008–09")
df["Year"] = df["Year"].str[:4].astype(int)


# 5. Strip whitespace from text columns


df["State"] = df["State"].str.strip()
df["Sex"] = df["Sex"].str.strip()


# 6. Remove unwanted rows 


# Drop rows where both Number and Rate are missing
df = df.dropna(subset=["Number", "Rate"], how="all")


# 7. Reset index


df.reset_index(drop=True, inplace=True)


# 8. Save cleaned dataset

print(f"Cleaned dataset shape: {df.shape}")

df.to_csv("preprocessed_data.csv", index=False)

print("Dataset cleaned successfully!")
