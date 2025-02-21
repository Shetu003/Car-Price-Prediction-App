import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load the dataset
file_path = 'car.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Kms_Driven'] = pd.to_numeric(df['Kms_Driven'], errors='coerce')
df.dropna(inplace=True)

df['Car_Age'] = 2025 - df['Year']
df.drop(['Year'], axis=1, inplace=True)

categorical_columns = ['Fuel_Type', 'Seller_Type', 'Transmission']
numerical_columns = ['Car_Age', 'Present_Price', 'Kms_Driven', 'Owner']

X = df[categorical_columns + numerical_columns]
y = df['Selling_Price']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
pipeline.fit(X_train, y_train)

# Save the trained model
joblib.dump(pipeline, 'car_price_model.pkl')
print("Model saved as 'car_price_model.pkl'")
