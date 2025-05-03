import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the data
print("Loading data...")
mail_data = pd.read_csv('mail_data.csv')
mail_data = mail_data.where((pd.notnull(mail_data)), '')

# Prepare the data
X = mail_data['Message']
y = mail_data['Category'].map({'spam': 1, 'legitimate': 0})

# Create and fit the vectorizer
print("Training vectorizer...")
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Train the model
print("Training model...")
model = LogisticRegression()
model.fit(X_transformed, y)

# Save the model and vectorizer
print("Saving model and vectorizer...")
joblib.dump(model, 'ai_detection_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Done!") 