
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the diabetes dataset"""
    logger.info("Loading and preprocessing data...")
    
    try:
        df = pd.read_csv("diabetes.csv")
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        
        # Data quality analysis
        logger.info("Data quality analysis:")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Data types:\n{df.dtypes}")
        
        # Handle zero values (which represent missing data in this dataset)
        cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        
        logger.info("Handling zero values (missing data):")
        for col in cols_with_zeros:
            zero_count = (df[col] == 0).sum()
            logger.info(f"{col}: {zero_count} zero values ({zero_count/len(df)*100:.1f}%)")
            
            # Replace zeros with median for better imputation
            median_val = df[df[col] != 0][col].median()
            df[col] = df[col].replace(0, median_val)
            logger.info(f"{col}: Replaced zeros with median value {median_val:.2f}")
        
        # Feature engineering
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 25, 30, float('inf')], 
                                   labels=[0, 1, 2, 3])  # Underweight, Normal, Overweight, Obese
        
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 30, 45, 65, float('inf')], 
                                labels=[0, 1, 2, 3])  # Young, Adult, Middle-aged, Senior
        
        df['Glucose_Category'] = pd.cut(df['Glucose'], 
                                       bins=[0, 100, 126, float('inf')], 
                                       labels=[0, 1, 2])  # Normal, Pre-diabetic, Diabetic
        
        # Create interaction features
        df['BMI_Age_Interaction'] = df['BMI'] * df['Age'] / 100
        df['Glucose_BMI_Interaction'] = df['Glucose'] * df['BMI'] / 100
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise

def train_and_evaluate_models(df):
    """Train multiple models and select the best one"""
    logger.info("Training and evaluating models...")
    
    # Prepare features and target
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                   'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
                   'BMI_Category', 'Age_Group', 'Glucose_Category',
                   'BMI_Age_Interaction', 'Glucose_BMI_Interaction']
    
    X = df[feature_cols]
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model configurations
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42),
            'params': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
        }
    }
    
    best_model = None
    best_score = 0
    best_model_name = ""
    model_results = {}
    
    for name, config in models.items():
        logger.info(f"Training {name}...")
        
        # Grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            config['model'], 
            config['params'], 
            cv=5, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Evaluate the best model
        best_estimator = grid_search.best_estimator_
        y_pred = best_estimator.predict(X_test_scaled)
        y_pred_proba = best_estimator.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_estimator, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        model_results[name] = {
            'model': best_estimator,
            'accuracy': accuracy,
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        logger.info(f"{name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  AUC Score: {auc_score:.4f}")
        logger.info(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        logger.info(f"  Best Params: {grid_search.best_params_}")
        
        # Select best model based on AUC score
        if auc_score > best_score:
            best_score = auc_score
            best_model = best_estimator
            best_model_name = name
    
    logger.info(f"Best model: {best_model_name} with AUC score: {best_score:.4f}")
    
    # Generate detailed evaluation for the best model
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Classification report
    logger.info("Classification Report for Best Model:")
    logger.info(f"\n{classification_report(y_test, y_pred_best)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_best)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Feature importance (if available)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info("Feature Importance:")
        logger.info(f"\n{feature_importance}")
    elif hasattr(best_model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': best_model.coef_[0]
        }).sort_values('coefficient', ascending=False, key=abs)
        logger.info("Feature Coefficients:")
        logger.info(f"\n{feature_importance}")
    
    return best_model, scaler, model_results, X_test_scaled, y_test

def save_model_artifacts(model, scaler, model_results):
    """Save model, scaler, and metadata"""
    logger.info("Saving model artifacts...")
    
    try:
        # Save model and scaler
        pickle.dump(model, open("logistic_model.pkl", "wb"))
        pickle.dump(scaler, open("scaler.pkl", "wb"))
        
        # Save model metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_results': {
                name: {
                    'accuracy': float(results['accuracy']),
                    'auc_score': float(results['auc_score']),
                    'cv_mean': float(results['cv_mean']),
                    'cv_std': float(results['cv_std']),
                    'best_params': results['best_params']
                }
                for name, results in model_results.items()
            }
        }
        
        with open("model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model artifacts saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model artifacts: {str(e)}")
        raise

def generate_model_report(model_results, X_test, y_test):
    """Generate a comprehensive model report"""
    logger.info("Generating model report...")
    
    try:
        # Create visualizations
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        ax1 = axes[0, 0]
        for name, results in model_results.items():
            model = results['model']
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {results['auc_score']:.3f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves Comparison')
        ax1.legend()
        ax1.grid(True)
        
        # Model Comparison
        ax2 = axes[0, 1]
        model_names = list(model_results.keys())
        accuracies = [model_results[name]['accuracy'] for name in model_names]
        auc_scores = [model_results[name]['auc_score'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax2.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        ax2.bar(x + width/2, auc_scores, width, label='AUC Score', alpha=0.8)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Score')
        ax2.set_title('Model Performance Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Best model confusion matrix
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
        best_model = model_results[best_model_name]['model']
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        ax3 = axes[1, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'Confusion Matrix - {best_model_name}')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Cross-validation scores
        ax4 = axes[1, 1]
        cv_means = [model_results[name]['cv_mean'] for name in model_names]
        cv_stds = [model_results[name]['cv_std'] for name in model_names]
        
        ax4.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Cross-Validation Score')
        ax4.set_title('Cross-Validation Performance')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_evaluation_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model report generated and saved as 'model_evaluation_report.png'")
        
    except Exception as e:
        logger.error(f"Error generating model report: {str(e)}")

def main():
    """Main training pipeline"""
    logger.info("Starting model training pipeline...")
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Train and evaluate models
        best_model, scaler, model_results, X_test, y_test = train_and_evaluate_models(df)
        
        # Save model artifacts
        save_model_artifacts(best_model, scaler, model_results)
        
        # Generate model report
        generate_model_report(model_results, X_test, y_test)
        
        logger.info("Model training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("MODEL TRAINING SUMMARY")
        print("="*50)
        for name, results in model_results.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  AUC Score: {results['auc_score']:.4f}")
            print(f"  CV Score: {results['cv_mean']:.4f} (+/- {results['cv_std']*2:.4f})")
        
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['auc_score'])
        print(f"\nBest Model: {best_model_name}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
