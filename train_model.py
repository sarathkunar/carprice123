# ==========================================
# HIGH ACCURACY AI CAR PRICE PREDICTION
# XGBOOST MODEL
# ==========================================

import pandas as pd
import numpy as np
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("cars.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# IMPORTANT FEATURES
# ==========================================

features = [

    'Make',
    'Model',
    'Variant',
    'Fuel_Type',
    'Body_Type',
    'Power',
    'Torque',
    'Engine_Type',
    'City_Mileage',
    'Gears',
    'Wheelbase',
    'Ground_Clearance',
    'Seating_Capacity',
    'Airbags',
    'Cruise_Control',
    'Android_Auto',
    'Apple_CarPlay',
    'Turbocharger',
    'Battery',
    'Electric_Range'

]

target = 'Ex-Showroom_Price'

# ==========================================
# KEEP REQUIRED COLUMNS
# ==========================================

df = df[features + [target]]

# ==========================================
# CLEAN PRICE
# ==========================================

def clean_price(x):

    try:

        x = str(x)

        x = x.replace('Rs.', '')
        x = x.replace(',', '')
        x = x.strip()

        return float(x)

    except:

        return np.nan

df[target] = df[target].apply(clean_price)

# ==========================================
# EXTRACT NUMBERS FROM TEXT
# ==========================================

def extract_number(value):

    try:

        value = str(value)

        number = re.findall(r"[-+]?(?:\\d*\\.\\d+|\\d+)", value)

        if len(number) > 0:

            return float(number[0])

        else:

            return np.nan

    except:

        return np.nan

# ==========================================
# CLEAN POWER & TORQUE
# ==========================================

df['Power'] = df['Power'].apply(extract_number)

df['Torque'] = df['Torque'].apply(extract_number)

# ==========================================
# NUMERIC COLUMNS
# ==========================================

numeric_columns = [

    'Power',
    'Torque',
    'City_Mileage',
    'Gears',
    'Wheelbase',
    'Ground_Clearance',
    'Seating_Capacity',
    'Airbags',
    'Electric_Range'

]

for col in numeric_columns:

    df[col] = pd.to_numeric(df[col], errors='coerce')

# ==========================================
# REMOVE NULL TARGET
# ==========================================

df = df.dropna(subset=[target])

# ==========================================
# FEATURES & TARGET
# ==========================================

X = df[features]

y = df[target]

# ==========================================
# CATEGORICAL & NUMERIC
# ==========================================

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# ==========================================
# PREPROCESSING
# ==========================================

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ]

)

# ==========================================
# XGBOOST MODEL
# ==========================================

model = Pipeline(steps=[

    ('preprocessor', preprocessor),

    ('regressor', XGBRegressor(

        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42

    ))

])

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# ==========================================
# TRAIN MODEL
# ==========================================

print("Training High Accuracy Model...")

model.fit(X_train, y_train)

print("Training Completed!")

# ==========================================
# PREDICTION
# ==========================================

predictions = model.predict(X_test)

# ==========================================
# EVALUATION
# ==========================================

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\n================================")

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)

print("Accuracy:", round(r2 * 100, 2), "%")

print("================================")

# ==========================================
# SAVE MODEL
# ==========================================

pickle.dump(model, open("model.pkl", "wb"))

print("High Accuracy Model Saved!")