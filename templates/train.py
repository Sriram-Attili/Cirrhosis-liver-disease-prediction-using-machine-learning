import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load your dataset
df = pd.read_csv("cirrhosis.csv")

# Define target
target = 'Status'
X = df.drop(columns=[target])

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Create the preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# Fit the preprocessor
preprocessor.fit(X)

# Save the preprocessor
with open("preprocessing.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

print("✅ preprocessing.pkl created successfully.")
