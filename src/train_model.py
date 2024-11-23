# src/train_model.py
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import logging
from prepare_data import load_data, prepare_data
import warnings
import pandas as pd
from datetime import datetime
from evaluate_model import evaluate_model

# Filter warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
def setup_logging():
    """Configure logging to both file and console"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def train_model(X_train, X_test, y_train, y_test):
    """
    Train XGBoost regression model with updated configuration
    """
    # Create parameter dictionary
    params = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 20,
        'eval_metric': ['rmse', 'mae']  # Move eval_metric to params
    }
    
    # Initialize model
    model = xgb.XGBRegressor(**params)
    
    # Create evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    
    # Train model
    logging.info("Starting model training...")
    model.fit(
        X_train, 
        y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    logging.info(f"Best iteration: {model.best_iteration}")
    logging.info(f"Best score: {model.best_score}")
    
    return model

def main():
    # Setup logging
    setup_logging()
    logging.info("Starting training process...")
    
    try:
        # Create directories if they don't exist
        for directory in ['data', 'models', 'outputs']:
            os.makedirs(directory, exist_ok=True)
            logging.info(f"Checked/created directory: {directory}")
        
        # Load and prepare data
        logging.info("Loading data...")
        df = load_data('data/dataset_cleaned.csv')
        
        logging.info("Preparing data...")
        df_clean = prepare_data(df)
        
        # Split data
        logging.info("Splitting data into train and test sets...")
        X = df_clean.drop('y', axis=1)
        y = df_clean['y']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log data shapes
        logging.info(f"Training set shape: {X_train.shape}")
        logging.info(f"Testing set shape: {X_test.shape}")
        
        # Train model
        model = train_model(X_train, X_test, y_train, y_test)

        # Evaluate model
        logging.info("Evaluating model...")
        y_pred = evaluate_model(model, X_test, y_test)  # This calculates MSE and R²

        
        # Save model
        model_path = 'models/xgboost_model.json'
        model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Save feature names
        feature_names = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_names.to_csv('outputs/feature_importance.csv', index=False)
        logging.info("Feature importance saved to outputs/feature_importance.csv")
        
        logging.info("Training process completed successfully!")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()