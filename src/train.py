import duckdb
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from pathlib import Path

import optuna
import xgboost as xgb
import lightgbm as lgbm
import catboost as cb
import numpy as np

from src.extract import extract_data
from src.load import load
from src.transform import preprocess
from src.transform import get_data

class TrainRegressors:
    def __init__(self, preprocess, model_name, cat_cols=None, num_cols=None):
        self.preprocess_pipeline = preprocess
        self.model_name = model_name
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def create_model(self, trial):
        if self.model_name == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'booster': 'gbtree',
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
            }
            params['enable_categorical'] = True

            return xgb.XGBRegressor(**params)
        
        elif self.model_name == 'lgbm': 
            params = {
                'objective': 'regression',
                'metric': 'mean_squared_error',
                'verbosity': -1,
                'boosting_type': 'gbdt',
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                'max_depth': trial.suggest_int('max_depth', -1, 50),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
                'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 10.0, log=True),
            }

            return lgbm.LGBMClassifier(**params)

        elif self.model_name == 'cb':
            params = {
                'loss_function': 'RMSE',
                'iterations': 1000,
                'verbose': False,
                'early_stopping_rounds': 50,
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 100.0, log=True),
                'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 1, 15),
            }
            return cb.CatBoostClassifier(**params)

        elif self.model_name == "rf":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 1000),  # Number of trees in the forest
                'max_depth': trial.suggest_int('max_depth', 3, 20),  # Maximum depth of the tree
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),  # Minimum samples required to split a node
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),  # Minimum samples required at a leaf node
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Number of features to consider for splitting
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),  # Whether to bootstrap samples
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced'])  # Class weights for imbalanced data
            }

            return Pipeline(steps=[
            ('preprocessor', self.preprocess_pipeline),
            ('regressor', RandomForestClassifier(**params))
            ])


        elif self.model_name == "isolation_trees":
            params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step = 100),
            'contamination': trial.suggest_float('contamination', 0.0, 0.5),
            'max_features' : trial.suggest_int('max_features', 1, 29),
            'bootstrap' : trial.suggest_categorical('bootstrap', [True, False]),
            'warm_start' : trial.suggest_categorical('warm_start', [True, False])
            }

            return Pipeline(steps=[
            ('preprocessor', self.preprocess_pipeline),
            ('regressor', (**params))
            ])
    def objective(self, trial, X, y):
        with mlflow.start_run(nested=True): 
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = []
            best_iterations = []

            mlflow.log_param("model_name", self.model_name)

            for train_idx, val_idx in kf.split(X, y):
                model = self.create_model(trial)
                #mlflow.log_param("model_name", model_name)

                # Split data into training and validation sets
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                if self.model_name == 'cb':
                    model.fit(X_train, y_train, cat_features=self.cat_cols.tolist())
                else:
                    model.fit(X_train, y_train)

                preds = model.predict(X_val)

                rmse = np.sqrt(mean_squared_error(y_val, preds))
                cv_scores.append(rmse)

            final_score = np.mean(cv_scores)

            if self.model_name == 'xgb':
                pip_requirements = [f"xgboost=={xgb.__version__}"]
                sig_input = X_train.head(5).copy()
                
                for col in self.cat_cols:
                    sig_input[col] = sig_input[col].astype(str)
                
                for col in self.num_cols:
                    sig_input[col] = sig_input[col].astype(float)


                signature = infer_signature(sig_input, pd.Series(preds[:5], name="prediction"))
            
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="model", 
                    pip_requirements=pip_requirements,
                    signature=signature
                )
            elif self.model_name == 'lgbm':
                print("WE ARE HERER")
                pip_requirements = [f"lightgbm=={lgbm.__version__}"]
                sig_input = X_train.head(5).copy()
                
                for col in self.cat_cols:
                    sig_input[col] = sig_input[col].astype(str)
                
                for col in self.num_cols:
                    sig_input[col] = sig_input[col].astype(float)


                signature = infer_signature(sig_input, pd.Series(preds[:5], name="prediction"))
            
                mlflow.lightgbm.log_model(
                    lgb_model=model,
                    name="model", 
                    pip_requirements=pip_requirements,
                    signature=signature
                )
            else:
                print("WE ARE HERER")
                input_example = self.input_ml(X_train) #input sample for mlflow model logging
                mlflow.sklearn.log_model(model, "model", name="model")

            mlflow.log_metric("rmse", final_score)
            

        return final_score
    
    def input_ml(self,data):
        input_example = data.head(5)

        for col in self.cat_cols:
             input_example[col] = input_example[col].astype(str)
        
        for col in self.num_cols:
            input_example[col] = input_example[col].astype(float)
        
        return input_example
    

if __name__ == "__main__":
    #ETL
    extract_data()
    load()
    X, y, X_test, y_test = get_data()
    preprocess_pipeline = preprocess(X)

    cat_cols = X.select_dtypes(include='category').columns
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns

    model_name = 'cb' 
    model_reg = TrainRegressors(preprocess_pipeline, model_name, cat_cols, num_cols)

    #study = model_reg.train(X, y, n_trials=2)
    mlflow.set_experiment("optuna-trained-model")

    with mlflow.start_run(run_name = model_name):
        study = optuna.create_study(direction="minimize", study_name=f"{model_name}_regressor_tuning")
        study.optimize(
            lambda trial: model_reg.objective(trial, X, y),
            n_trials=2,
        )

        # Log best result
        mlflow.log_metric("best_rmse", study.best_value)
        mlflow.log_params(study.best_params)

        
    

