# preprocess and feature enginering
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer



def get_data():
    df = pd.read_parquet("data/raw/creditcard_fraud_raw.parquet")

    X = train_df.drop(columns=['Class'])
    y = train_df['Class']
    
    X_train, X_trest, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)
    
    X = X_train.copy()
    y = y_train.copy()
    

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    for col in categorical_cols:
        X[col] = X[col].astype('category')
        test_df[col] = test_df[col].astype('category')

    return X, y, X_test, y_test

def preprocess(data):
    preprocess_columns = ['Time', 'Amount']preprocess_columns 


    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), preprocess_columns)   
    ],
    remainder='passthrough' # 'remainder' is now less critical, but good practice
    )

    return preprocessor