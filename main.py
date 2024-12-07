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




def unique_encode_column(df, column_name):
    """
    Encodes a column in the dataframe with unique numeric values for each unique category.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column_name (str): The column to be encoded.

    Returns:
    pd.DataFrame: The dataframe with the encoded column.
    dict: The mapping of original values to encoded values.
    """
    # Create a unique mapping for the column
    unique_values = df[column_name].unique()
    encoding_map = {value: idx for idx, value in enumerate(unique_values)}

    # Replace the column values with their encoded values
    df[column_name] = df[column_name].map(encoding_map)

    return df, encoding_map

columns_to_encode= list(df.columns)

all_encodings={}

for col in columns_to_encode:
    df, encoding_map = unique_encode_column(df, col)
    all_encodings[col] = encoding_map

# Example usage:

print(all_encodings)  # To see the updated dataframe

print(df.head())