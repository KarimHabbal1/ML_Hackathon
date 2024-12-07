import pandas as pd
import numpy as np 
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


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



# Standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)  # Assuming df is the dataset with 40 features



# Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)  # For 2D visualization
data_pca = pca.fit_transform(data_scaled)

# Optional: Check how much variance is explained by the components
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance by Components: {explained_variance}")




# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Choose the number of clusters (e.g., 3)
clusters1 = kmeans.fit_predict(data_pca)  # Cluster labels for each data point
# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)  # Choose the number of clusters (e.g., 3)
clusters2 = kmeans.fit_predict(data_pca)  # Cluster labels for each data point
# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)  # Choose the number of clusters (e.g., 3)
clusters3 = kmeans.fit_predict(data_pca)  # Cluster labels for each data point


# Scatter plot of the PCA-reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters1, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters (PCA-Reduced Data)')
plt.colorbar(label='Cluster Label')
plt.show()
# Scatter plot of the PCA-reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters2, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters (PCA-Reduced Data)')
plt.colorbar(label='Cluster Label')
plt.show()
# Scatter plot of the PCA-reduced data
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters3, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-Means Clusters (PCA-Reduced Data)')
plt.colorbar(label='Cluster Label')
plt.show()
