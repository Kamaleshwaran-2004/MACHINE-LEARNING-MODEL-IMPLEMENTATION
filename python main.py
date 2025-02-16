import pandas as pd
import numpy as np
import nltk
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords

nltk.download('stopwords')

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    df = df[['label', 'message']]  
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})  
    return df

def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'\d+', '', text)  
    text = text.translate(str.maketrans('', '', string.punctuation))  
    text = text.strip()  
    text_tokens = text.split()
    text_tokens = [word for word in text_tokens if word not in stopwords.words('english')] 
    return " ".join(text_tokens)

def train_and_evaluate(df):
    df['message'] = df['message'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    return vectorizer, model

def predict_message(vectorizer, model, new_message):
    new_message_processed = preprocess_text(new_message)
    new_message_tfidf = vectorizer.transform([new_message_processed])
    prediction = model.predict(new_message_tfidf)
    return "Spam" if prediction[0] == 1 else "Ham"

def main():
    file_path = "spam.csv"
    df = load_data(file_path)
    vectorizer, model = train_and_evaluate(df)

    test_message = "Congratulations! You've won a free gift."
    result = predict_message(vectorizer, model, test_message)
    print(f"Test Message: {test_message}")
    print(f"Prediction: {result}")

if __name__ == "__main__":
    main()
