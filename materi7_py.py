# %% [1] Loading library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# %% [2] Load Dataset (Menggunakan Iris Dataset)
iris = load_iris()
# Membuat DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Menampilkan 5 data teratas
print(df.head())
# Deskripsi statistik data 
print(df.describe())

# %% [3] Visualisasi Distribusi (Contoh: Sepal Length) 
sns.displot(df['sepal length (cm)'], kde=True)
plt.title("Distribution of Sepal Length")
plt.show()

# %% [4] Correlation Matrix 
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.title("Correlation Matrix")
plt.show()

# %% [5] Pembagian Data (Training dan Testing) 
# X = fitur, y = target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split data 75% train, 25% test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling 
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %% [6] Membuat Model Naive Bayes (GaussianNB) 
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# %% [7] Prediksi dan Evaluasi 
y_pred = classifier.predict(X_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# %% [8] Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# %% [9] Uji Coba Data Baru (Contoh data bunga baru) 
# Prediksi bunga dengan ukuran sepal/petal tertentu
new_data = [[5.1, 3.5, 1.4, 0.2]] # Ukuran dalam cm
prediction = classifier.predict(sc.transform(new_data))
print(f"Hasil Prediksi Spesies: {iris.target_names[prediction][0]}")
