from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# Load the UCI Heart Disease dataset
heart_disease = fetch_openml(name='heart')

# Access data
X = heart_disease.data
y = heart_disease.target

# Splitting the data into training and testing sets using Hold-out method
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imputation to handle NaNs
imputer = SimpleImputer(strategy='mean')  # Filling NaNs with the mean of existing values
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Initializing and training MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
mlp.fit(X_train_imputed, y_train)

# Predicting on test set
predictions = mlp.predict(X_test_imputed)

# Calculating accuracy and confusion matrix
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

indices = np.arange(len(y_test))

plt.figure(figsize=(8, 6))
plt.scatter(indices, y_test, color='blue', label='Valores Reais', alpha=0.5)
plt.scatter(indices, predictions, color='red', label='Valores Preditos', alpha=0.5)
plt.xlabel('Índices dos Exemplos')
plt.ylabel('Valores')
plt.title('Valores Reais vs Preditos')
plt.legend()
plt.show()

print("Precisão:", accuracy)
print(" Matriz Confusão:\n", conf_matrix)
