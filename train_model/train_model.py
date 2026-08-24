import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
df = pd.read_csv("cars.csv")

print("Dataset Shape:", df.shape)

# Selected Features
features = [
    'Model',
    'Variant',
    'Fuel_Type',
    'City_Mileage',
    'Gears',
    'Front_Brakes',
    'Rear_Brakes',
    'Wheelbase',
    'Power_Steering',
    'Engine_Type',
    'Cruise_Control',
    'Android_Auto',
    'Rain_Sensing_Wipers',
    'Automatic_Headlamps',
    'Battery',
    'Electric_Range'
]

target = 'Ex-Showroom_Price'

# Keep required columns
df = df[features + [target]]

# Clean Price Column
def clean_price(x):
    try:
        x = str(x).replace('Rs.', '').replace(',', '').strip()
        return float(x)
    except:
        return np.nan

df[target] = df[target].apply(clean_price)

# Remove null target
df = df.dropna(subset=[target])

# Convert numeric columns
numeric_cols = [
    'City_Mileage',
    'Gears',
    'Wheelbase',
    'Electric_Range'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Features and Target
X = df[features]
y = df[target]

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Numeric Transformer
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical Transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Model Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))

print("Model Saved Successfully!")