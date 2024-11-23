# src/prepare_data.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """
    Load data from CSV file
    """
    return pd.read_csv(file_path)

def prepare_data(df):
    """
    Prepare the data for regression by handling missing values,
    encoding categorical variables, and scaling numeric features
    """
    df_clean = df.copy()
    
    # Identify numeric and categorical columns
    numeric_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    
    # Handle missing values
    for col in numeric_columns:
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    for col in categorical_columns:
        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        df_clean[col] = le.fit_transform(df_clean[col])
    
    # Scale numeric features (excluding target 'y')
    scaler = StandardScaler()
    numeric_features = [col for col in numeric_columns if col != 'y']
    df_clean[numeric_features] = scaler.fit_transform(df_clean[numeric_features])
    
    return df_clean