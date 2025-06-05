# machine-learning-lab
prg1

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load dataset
df = pd.DataFrame(fetch_california_housing().data, columns=fetch_california_housing().feature_names)
df['Target'] = fetch_california_housing().target

# Histograms
df.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle('Histograms')
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Box Plots
plt.figure(figsize=(12, 10))
sns.boxplot(data=df, orient='h', palette='Set2')
plt.title('Box Plots')
plt.show()
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2. prgm

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the dataset
df = pd.DataFrame(fetch_california_housing().data, columns=fetch_california_housing().feature_names)
df['target'] = fetch_california_housing().target

# Correlation matrix heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pair plot
sns.pairplot(df)
plt.suptitle('Pairwise Relationships', y=1.02)
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


3.pgrm

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load and standardize the Iris dataset
X = load_iris().data
X_scaled = StandardScaler().fit_transform(X)

# Compute the covariance matrix and perform eigendecomposition
cov_matrix = np.cov(X_scaled.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Project data onto the top 2 principal components
X_pca = X_scaled.dot(eigenvectors[:, :2])

# Plot the results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=load_iris().target, cmap='viridis')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA of Iris Dataset')
plt.colorbar(label='Target Class')
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


4. pgrm

import csv

h=['0'for i in range(6)]
with open("C:\\Users\\SKSVMACET\\Desktop\\kl.csv") as f:
    data=csv.reader(f)
    data=list(data)
    
    for i in data:
        if i[-1]=="Yes":
            for j in range(6):
                if h[j]=='0':
                    h[j]=i[j]
                elif h[j]!=i[j]:
                    h[j]='?'

print(h)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



5. prgm

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Generate 100 random points between 0 and 1
X = np.random.rand(100, 1)
y = np.array([1 if x <= 0.5 else 2 for x in X[:50]])

# Classify remaining points (X[50:] with unknown labels)
X_new = X[50:]

# k-NN classification for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]
plt.figure(figsize=(10, 8))

for i, k in enumerate(k_values, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X[:50], y)
    y_pred = knn.predict(X_new)

    plt.subplot(3, 3, i)
    plt.scatter(X[:50], y, c=y, marker='o')
    plt.scatter(X_new, y_pred, c=y_pred, marker='x')
    plt.title(f'k={k}')
    plt.xlabel('x')
    plt.ylabel('Class')

plt.tight_layout()
plt.show()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



6. prgm

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
X = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(100)

# Perform Locally Weighted Regression
def lwr(X, y, query_points, tau=0.1):
    predictions = []
    for x in query_points:
        W = np.diag([np.exp(-np.linalg.norm(x - xi) ** 2 / (2 * tau ** 2)) for xi in X])
        X_bias = np.hstack((np.ones((len(X), 1)), X))
        theta = np.linalg.pinv(X_bias.T @ W @ X_bias) @ (X_bias.T @ W @ y)
        predictions.append(np.hstack(([1], x)) @ theta)
    return predictions

# Visualize data and LWR result
query_points = np.linspace(0, 1, 100).reshape(-1, 1)
predictions = lwr(X, y, query_points)

plt.scatter(X, y, color='blue')
plt.plot(query_points, predictions, color='red', lw=2)
plt.title('Locally Weighted Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


7. prgm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_california():
    housing = fetch_california_housing()
    X, y = housing.data[:, 3].reshape(-1, 1), housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue")
    plt.plot(X_test, y_pred, color="red")
    plt.xlabel("AveRooms")
    plt.ylabel("House Price ($100K)")
    plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

def polynomial_regression_auto_mpg():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    data = pd.read_csv(url, sep='\s+', names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"], na_values="?").dropna()
    X, y = data["displacement"].values.reshape(-1, 1), data["mpg"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(X_train, y_train)
    y_pred = model.predict(X_test)

    plt.scatter(X_test, y_test, color="blue")
    plt.scatter(X_test, y_pred, color="red")
    plt.xlabel("Displacement")
    plt.ylabel("MPG")
    plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred), "R2:", r2_score(y_test, y_pred))

if __name__ == "__main__":
    linear_regression_california()
    polynomial_regression_auto_mpg()

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


8. prgm


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load the Breast Cancer dataset
data = load_breast_cancer()
X = data.data  # Features
y = data.target  # Labels (0 for malignant, 1 for benign)

# Step 2: Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate the model on the test data
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Decision Tree classifier: {accuracy:.2f}")

# Step 5: Classify a new sample (using an example sample from the dataset)
# Let's use the first sample from the test set as an example for prediction
sample = X_test[0].reshape(1, -1)  # Reshaping the sample to 2D for prediction

# Predicting the class of the new sample
predicted_class = clf.predict(sample)
class_names = data.target_names

# Output the predicted class for the sample
print(f"The new sample is classified as: {class_names[predicted_class[0]]}")



---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


9. prgm

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load the Olivetti Faces dataset
data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data
y = data.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# Make predictions on the test set
y_pred = gnb.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Print the classification report (without zero_division argument)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation accuracy
cross_val_accuracy = cross_val_score(gnb, X, y, cv=5, scoring='accuracy')
print(f'\nCross-validation accuracy: {cross_val_accuracy.mean() * 100:.2f}%')

# Plotting some sample images from the test set
fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, image, label, prediction in zip(axes.ravel(), X_test, y_test, y_pred):
    ax.imshow(image.reshape(64, 64), cmap=plt.cm.gray)
    ax.set_title(f"True: {label}, Pred: {prediction}")
    ax.axis('off')

plt.show()


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




10. prgm 


import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
X = load_breast_cancer().data
X = StandardScaler().fit_transform(X)

# Apply KMeans clustering
y_kmeans = KMeans(n_clusters=2, random_state=42).fit_predict(X)

# PCA for 2D visualization
X_pca = PCA(n_components=2).fit_transform(X)

# Plot the clustering result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', s=50, alpha=0.6)
plt.title('K-Means Clustering on Breast Cancer Dataset')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
