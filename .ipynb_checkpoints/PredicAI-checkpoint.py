import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans  # Clustering
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Data Preprocessing
from sklearn.model_selection import train_test_split  # Train-test split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report  # Model evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("Mall_Customers.csv")  # Load dataset
print(df.info())  # Check for missing values
print(df.describe())  # Summary statistics


# Initialize LabelEncoder
encoder = LabelEncoder()

# Apply encoding to the 'Genre' column
df['Genre'] = encoder.fit_transform(df['Genre'])



X = df[['Annual Income (k$)', 'Spending Score (1-100)']]  # Select features
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Clusters")
plt.show()
df['Cluster'] = kmeans.fit_predict(X)


# Convert Spending Score into Two Categories (Binary Classification)
df['Spending Category'] = df['Spending Score (1-100)'].apply(lambda x: 1 if x > 50 else 0)

# Features & Target
X = df[['Age', 'Annual Income (k$)']]
y = df['Spending Category']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model (Binary Classification)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict Spending Categories
y_pred = model.predict(X_test)

# Generate Confusion Matrix
cm = confusion_matrix(y_test, y_pred)




# Plot Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("2×2 Confusion Matrix")
plt.savefig("static/confusion_matrix.png")
plt.show()
# Print Classification Report
print(classification_report(y_test, y_pred))

#plot cluster
df['Cluster'].value_counts().plot(kind='bar', color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.title("Customer Distribution Across Clusters")
plt.savefig("static/cluster_distribution.png")
plt.show()