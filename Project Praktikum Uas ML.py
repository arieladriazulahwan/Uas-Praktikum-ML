# Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Pastikan dataset bersih dari NaN di kolom target
df = pd.read_csv('creditcard.csv')
df = df.dropna(subset=['Class'])

# Pisahkan fitur dan target
X = df.drop('Class', axis=1)
y = df['Class']

# Lanjutkan train_test_split seperti sebelumnya
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Normalisasi fitur (kecuali 'Time' jika mau dikecualikan)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Tangani imbalance dengan SMOTE (oversampling)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# 6. Training model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_res, y_train_res)


# 7. Prediksi dan evaluasi
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cek proporsi kelas fraud dan normal
print("Distribusi kelas:")
print(y.value_counts(normalize=True))
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title("Distribusi Kelas: Normal vs Fraud")
plt.show()


# 2. Statistik deskriptif fitur 'Amount' (jumlah transaksi)
print("\nStatistik fitur Amount:")
print(df['Amount'].describe())

plt.figure(figsize=(8,4))
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title("Distribusi Jumlah Transaksi (Amount)")
plt.show()

# 3. Visualisasi fitur 'Time' (waktu transaksi dalam detik sejak awal rekaman)
plt.figure(figsize=(8,4))
sns.histplot(df['Time'], bins=50, kde=True)
plt.title("Distribusi Waktu Transaksi (Time)")
plt.show()


# 4. Heatmap korelasi fitur dengan target 'Class'
plt.figure(figsize=(12,8))
corr = df.corr()
sns.heatmap(corr[['Class']].sort_values(by='Class', ascending=False), annot=True, cmap='coolwarm')
plt.title("Korelasi Fitur dengan Kelas Fraud")
plt.show()

# 5. Visualisasi Confusion Matrix hasil model
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, cmap='Blues')
plt.title("Confusion Matrix Model Random Forest")
plt.show()


# 6. Visualisasi ROC Curve
from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_estimator(model, X_test_scaled, y_test)
plt.title("ROC Curve Model Random Forest")
plt.show()
