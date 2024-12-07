import pandas as pd
import numpy as np 
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


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


print(all_encodings) 

print(df.head())



#Standardize the dataset
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df)  



#Apply PCA to reduce to 2 components for visualization
pca = PCA(n_components=2)  #For 2D visualization
data_pca = pca.fit_transform(data_scaled)


#Elbow Method: Compute inertia for different values of k
inertia = []
k_values = range(1, 11)  #Testing cluster sizes from 1 to 10

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_pca)  #Fit the PCA-reduced data
    inertia.append(kmeans.inertia_)  #Append the inertia (sum of squared distances)

#Plot the Elbow Graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Find Optimal Number of Clusters')
plt.xticks(k_values)
plt.grid()
plt.show()


#Evaluate Silhouette Score for 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=42)
clusters_3 = kmeans_3.fit_predict(data_pca)
silhouette_3 = silhouette_score(data_pca, clusters_3)

#Evaluate Silhouette Score for 4 clusters
kmeans_4 = KMeans(n_clusters=4, random_state=42)
clusters_4 = kmeans_4.fit_predict(data_pca)
silhouette_4 = silhouette_score(data_pca, clusters_4)

#Print the Silhouette Scores
print(f"Silhouette Score for 3 Clusters: {silhouette_3:.3f}")
print(f"Silhouette Score for 4 Clusters: {silhouette_4:.3f}")

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_3, cmap='viridis', s=50, alpha=0.7)
plt.title('K-Means with 3 Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters_4, cmap='viridis', s=50, alpha=0.7)
plt.title('K-Means with 4 Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()





#SUPERVISED

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_pca, clusters_3, test_size=0.2, random_state=42)

#One-hot encode the cluster labels
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))


#Define the feed-forward neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  #Input layer with 16 neurons
    Dense(8, activation='relu'),  #Hidden layer with 8 neurons
    Dense(3, activation='softmax')  #Output layer with 3 neurons (one for each cluster)
])

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Display the model architecture
model.summary()

#Train the model
history = model.fit(X_train, y_train_encoded, epochs=50, batch_size=16, validation_split=0.2)

#Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=0)
print(f"Test Accuracy: {test_accuracy:.2f}")


#Predict clusters for test data
predicted_clusters = model.predict(X_test)
predicted_labels = predicted_clusters.argmax(axis=1)  # Convert one-hot predictions to cluster labels

#Compare predictions with actual labels
print(f"Predicted Labels: {predicted_labels}")
print(f"Actual Labels: {y_test}")


#Scatter plot of the predicted clusters
plt.scatter(X_test[:, 0], X_test[:, 1], c=predicted_labels, cmap='viridis', s=50, alpha=0.7)
plt.title('Predicted Clusters from Feed-Forward Neural Network')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()


#Predict cluster labels for the test set
predicted_clusters = model.predict(X_test)
predicted_labels = predicted_clusters.argmax(axis=1)  #Convert probabilities to cluster labels

#Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"Model Accuracy: {accuracy:.2f}")
