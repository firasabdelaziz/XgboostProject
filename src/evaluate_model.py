import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import numpy as np
import xgboost as xgb

def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model performance and create visualizations
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability for positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print metrics
    print("\nClassification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create visualizations
    create_evaluation_plots(y_test, y_pred, y_pred_proba, model)
    
    return y_pred

def create_evaluation_plots(y_test, y_pred, y_pred_proba, model):
    """
    Create evaluation plots for classification model
    """
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('outputs/roc_curve.png')
    plt.close()
    
    # Feature Importance
    plt.figure(figsize=(12, 6))
    xgb.plot_importance(model, max_num_features=20)
    plt.title('Feature Importance')
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    # Prediction Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=y_pred_proba, bins=50, kde=True)
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.savefig('outputs/prediction_distribution.png')
    plt.close()