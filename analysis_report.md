# Business Dataset Preprocessing — Analysis Report

## Dataset Overview

- **File**: `Business_dataset.csv`
- **Rows**: 4,362
- **Columns**: 5 — `Category`, `Price`, `Rating`, `Stock`, `Discount`
- **Data types**: 2 categorical (Category, Stock), 3 numeric (Price, Rating, Discount)

---

## Task 1: Missing Value Treatment

### Missing Value Analysis

| Column | Missing Count | Missing % | Strategy |
|---|---|---|---|
| Category | 2,830 | 64.9% | Mode imputation |
| Price | 64 | 1.5% | Median imputation |
| Rating | 1,039 | 23.8% | Mean imputation |
| Stock | 1,807 | 41.4% | Mode imputation |
| Discount | 622 | 14.3% | Median imputation |

### Methodology and Justification

**1. Category — Mode Imputation**
- **Method**: Filled missing values with the most frequent category.
- **Reason**: As a nominal categorical variable, no arithmetic operation (mean/median) is valid. Mode preserves the existing category distribution without introducing artificial categories.
- **Data quality impact**: Maintains class distribution integrity; avoids misclassification bias.

**2. Price — Median Imputation**
- **Method**: Filled missing values with the median price.
- **Reason**: Price contains a wide range (approx. 100–10,000) and likely has outliers. Median is robust to extreme values, unlike the mean which would be pulled by high-end prices.
- **Data quality impact**: Preserves central tendency without distortion from outliers.

**3. Rating — Mean Imputation**
- **Method**: Filled missing values with the mean rating.
- **Reason**: Rating values are bounded (~1–5) and appear roughly symmetrically distributed. Mean is the most efficient estimator for symmetric distributions with no severe outliers.
- **Data quality impact**: Provides the best unbiased estimate of the typical rating.

**4. Stock — Mode Imputation**
- **Method**: Filled missing values with the most frequent stock status.
- **Reason**: Binary categorical variable where mode simply assigns the majority class, which is the safest single guess for a categorical variable.
- **Data quality impact**: Preserves the majority-class proportion while filling unknowns.

**5. Discount — Median Imputation**
- **Method**: Filled missing values with the median discount.
- **Reason**: Discount values range from 0–49 with many zeros (no discount) and a right-skewed distribution. Median avoids the inflation that mean would cause from high discounts.
- **Data quality impact**: More representative of the typical discount value.

---

## Task 2: Data Normalization

### Technique Applied

**Min-Max Normalization** (also called Min-Max Scaling)

Formula:
```
X_scaled = (X - X_min) / (X_max - X_min)
```

### Columns Normalized

| Column | Original Range | After Normalization |
|---|---|---|
| Price | ~100 – 10,000 | [0, 1] |
| Rating | ~1.0 – 5.0 | [0, 1] |
| Discount | 0 – 49 | [0, 1] |

### Why Min-Max Normalization?

1. **Comparable scale**: All three numeric features are scaled to the [0,1] range, making their magnitudes directly comparable. Without normalization, Price (thousands) would dominate Rating (~1–5) in any distance-based or gradient-based algorithm.

2. **Preserves distribution shape**: Unlike standardization (z-score), Min-Max preserves the exact relative distances between data points. A product with twice the price of another will still have twice the scaled value.

3. **Bounded output**: Values are confined to [0,1], which is ideal for models that expect inputs in this range (neural networks, regularization-based models).

4. **Interpretability**: The scaled value directly represents where a sample falls within the observed range — e.g., a Price of 0.5 means it is halfway between the cheapest and most expensive product.

5. **No distributional assumptions**: Unlike z-score normalization (which assumes approximately normal data), Min-Max works regardless of the underlying distribution.

### Why Not Z-Score Standardization?

Z-score would center data at 0 with unit variance, but:
- It does not bound values — outliers can produce extreme z-scores
- Negative values are less intuitive to interpret
- It assumes at least roughly normal data, which Discount (with many zeros) violates

### Why Not Robust Scaling?

Robust scaling (using median and IQR) is useful when outliers are extreme, but the ranges in this dataset are well-behaved and bounded, so Min-Max is sufficient and more interpretable.

---

## Summary

| Step | Method | Purpose |
|---|---|---|
| Missing values (categorical) | Mode imputation | Preserves distribution |
| Missing values (numeric) | Median imputation | Robust to outliers |
| Missing values (rating) | Mean imputation | Efficient for symmetric data |
| Normalization | Min-Max scaling | Unifies scale to [0,1] |

All preprocessing decisions were made to maximize data quality, preserve statistical properties, and ensure downstream model compatibility.
