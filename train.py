import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ✅ Load and clean dataset
df = pd.read_csv("cirrhosis.csv")

# ✅ Drop rows with missing target values
df = df.dropna(subset=['Stage', 'Status'])

# ✅ Drop ID column
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Define targets
target_stage = 'Stage'
target_status = 'Status'
X = df.drop(columns=[target_stage, target_status])

# Identify numeric and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# Model pipeline for Stage
stage_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
stage_pipe.fit(X, df[target_stage])

# Model pipeline for Status
status_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier())
])
status_pipe.fit(X, df[target_status])

# Save both models
with open("model_stage.pkl", "wb") as f:
    pickle.dump(stage_pipe, f)

with open("model_status.pkl", "wb") as f:
    pickle.dump(status_pipe, f)

print("✅ Models saved: model_stage.pkl and model_status.pkl")
