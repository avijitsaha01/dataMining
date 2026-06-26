import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Business_dataset.csv")

print("=" * 60)
print("BUSINESS DATASET PREPROCESSING")
print("=" * 60)

print("\n--- INITIAL DATA INFO ---")
print(f"Shape: {df.shape}")
print(f"\nMissing values per column:")
print(df.isnull().sum())
print(f"\nMissing % per column:")
print(round(df.isnull().sum() / len(df) * 100, 2))

print("\n--- COLUMN DATA TYPES ---")
print(df.dtypes)

print("\n--- DESCRIPTIVE STATISTICS (BEFORE) ---")
print(df.describe(include='all').to_string())

print("\n\n=== TASK 1: MISSING VALUE TREATMENT ===\n")

# 1. Category - Mode imputation
category_mode = df['Category'].mode()[0]
print(f"1. Category: Mode imputation -> '{category_mode}'")
df['Category'] = df['Category'].fillna(category_mode)

# 2. Price - Median imputation (robust to outliers)
price_median = df['Price'].median()
print(f"2. Price: Median imputation -> {price_median}")
df['Price'] = df['Price'].fillna(price_median)

# 3. Rating - Mean imputation
rating_mean = round(df['Rating'].mean(), 4)
print(f"3. Rating: Mean imputation -> {rating_mean}")
df['Rating'] = df['Rating'].fillna(rating_mean)

# 4. Stock - Mode imputation
stock_mode = df['Stock'].mode()[0]
print(f"4. Stock: Mode imputation -> '{stock_mode}'")
df['Stock'] = df['Stock'].fillna(stock_mode)

# 5. Discount - Median imputation
discount_median = df['Discount'].median()
print(f"5. Discount: Median imputation -> {discount_median}")
df['Discount'] = df['Discount'].fillna(discount_median)

print(f"\nMissing values AFTER treatment:")
print(df.isnull().sum())

df_cleaned = df.copy()
df_cleaned.to_csv("Business_dataset_cleaned.csv", index=False)
print("\nCleaned dataset saved: Business_dataset_cleaned.csv")

print("\n\n=== TASK 2: DATA NORMALIZATION ===\n")

print("Technique: Min-Max Normalization (scaling to [0,1])")
print("Applied to: Price, Rating, Discount\n")

scaler = MinMaxScaler()
numeric_cols = ['Price', 'Rating', 'Discount']

print("Before Normalization:")
for col in numeric_cols:
    print(f"  {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}, std={df[col].std():.4f}")

df_norm = df.copy()
df_norm[numeric_cols] = scaler.fit_transform(df_norm[numeric_cols])

print("\nAfter Normalization:")
for col in numeric_cols:
    print(f"  {col}: min={df_norm[col].min():.4f}, max={df_norm[col].max():.4f}, mean={df_norm[col].mean():.4f}, std={df_norm[col].std():.4f}")

df_norm.to_csv("Business_dataset_normalized.csv", index=False)
print("\nNormalized dataset saved: Business_dataset_normalized.csv")

print("\n--- FINAL CLEANED DATA INFO ---")
print(f"Shape: {df_cleaned.shape}")
print(df_cleaned.head(10).to_string())

print("\n--- FINAL NORMALIZED DATA SAMPLE (first 10 rows) ---")
print(df_norm.head(10).to_string())

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
