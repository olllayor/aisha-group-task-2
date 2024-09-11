import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
import re
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
# NLTK ma'lumotlarini yuklash

nltk.download('stopwords')

# Ma'lumotlarni yuklash
df = pd.read_csv('archive/train.csv', names=['polarity', 'title', 'text'])

# Birinchi qatorlarni va ma'lumotlar haqida asosiy ma'lumotlarni ko'rsatish

print(df.head())

# print(df.info())

# Matnni qayta ishlash
stop_words = set(stopwords.words('english'))

# print(stop_words)

def preprocess_text(text):
    # Kichik harfga o'tkazish
    text = text.lower()
    # Ishoralarni o'chirish 
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Stopwordsni o'chirish
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# 'text' ustuniga qayta ishlangan matnni qo'shish
df['processed_text'] = df['text'].apply(preprocess_text)

print(df.head())

# Label'larni kodilash
df['label'] = df['polarity'].map({1: 0, 2: 1})

# X (o'zgarishlar) va y (maqsad) ni ajratish
X = df['processed_text']
y = df['label']

# Ma'lumotlarni trening va sinov setlariga bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data has been preprocessed.")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Matnni vektorlashtirish
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Modelni trening qilish
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_vectorized, y_train)

# Tahmin qilish
y_pred = model.predict(X_test_vectorized)

# Modelni baholash
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Qarash matrisini chiqarish
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

