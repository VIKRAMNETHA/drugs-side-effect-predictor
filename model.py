"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load the dataset
data = pd.read_csv('drugs.csv')

# Split the data
X = data['Drug Name']
y = data['Side Effect']

# Vectorize the drug names
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and vectorizer for future use
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
"""