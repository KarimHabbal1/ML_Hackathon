import pandas as pd
import numpy as np 
from scipy.stats import chi2_contingency


df = pd.read_excel('Dataset.xls')
df.info()

df = df.drop(columns=['Unnamed: 0','Last page', 'What is your current marital status? [Comment]', 'What is your current employment status? [Comment]' , 'What is your main source of income? [Comment]', 'What type of income or financial support does your household receive? [Comment]'])

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate the correlation matrix
corr_matrix = numeric_df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find redundant columns
redundant_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

# Drop redundant columns
df = df.drop(columns=redundant_cols)
print("Dropped columns:", redundant_cols)


# Function to compute Cramér's V
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Set a threshold for high association
threshold = 0.95

# Store columns to drop
columns_to_drop = set()

# Iterate over all pairs of categorical columns
for i, col1 in enumerate(categorical_cols):
    for col2 in categorical_cols[i + 1:]:  # Avoid redundant pairs
        if col1 not in columns_to_drop and col2 not in columns_to_drop:
            cramers_value = cramers_v(df[col1], df[col2])
            print(f"Cramér's V between {col1} and {col2}: {cramers_value}")

            # Drop one column if the association is high
            if cramers_value > threshold:
                print(f"Dropping column: {col2} (high association with {col1})")
                columns_to_drop.add(col2)

# Drop redundant columns from the DataFrame
df = df.drop(columns=list(columns_to_drop))

print("Remaining columns:", df.columns)

print(df.head())