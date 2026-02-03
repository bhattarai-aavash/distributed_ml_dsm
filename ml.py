import pandas as pd
import torch

print(torch.cuda.is_available())
# Path to your parquet file
file_path = "/home/abhattar/ml_cars_main/cic_split/CIC_training_set.csv"

# Read parquet file into DataFrame
df = pd.read_csv(file_path)  # or engine="fastparquet"

# Show first few rows
# print(df.head())
print(df.info())

print(df.describe(include='all')) 

print("Unique labels:", df["Label"].unique())
print("\nData types:\n", df.dtypes)     