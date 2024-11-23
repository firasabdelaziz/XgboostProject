import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance and create visualizations
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create visualizations
    create_evaluation_plots(y_test, y_pred, model)
    
    return y_pred

def create_evaluation_plots(y_test, y_pred, model):
    """
    Create evaluation plots
    """
    # Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig('outputs/actual_vs_predicted.png')
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # Error Distribution
    errors = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(errors, kde=True)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.savefig('outputs/error_distribution.png')
    plt.close()