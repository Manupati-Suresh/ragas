# Model explanation and interpretability module

import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ModelExplainer:
    """Enhanced class for comprehensive model explanation and interpretability"""
    
    def __init__(self, model_path: str = "logistic_model.pkl", scaler_path: str = "scaler.pkl"):
        """
        Initialize the ModelExplainer
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self.feature_descriptions = {
            'Pregnancies': 'Number of times pregnant',
            'Glucose': 'Plasma glucose concentration (mg/dL)',
            'BloodPressure': 'Diastolic blood pressure (mm Hg)',
            'SkinThickness': 'Triceps skin fold thickness (mm)',
            'Insulin': '2-Hour serum insulin (μU/mL)',
            'BMI': 'Body mass index (kg/m²)',
            'DiabetesPedigreeFunction': 'Genetic predisposition score',
            'Age': 'Age in years'
        }
        self.load_model()
    
    def load_model(self) -> bool:
        """
        Load the trained model and scaler with enhanced error handling
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                logger.warning("Model files not found. Please train the model first.")
                return False
                
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            logger.info("Model and scaler loaded successfully for explanation")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model for explanation: {str(e)}")
            return False
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get comprehensive feature importance from the logistic regression model
        
        Returns:
            DataFrame with feature importance metrics
        """
        if self.model is None:
            return pd.DataFrame()
        
        try:
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                importance = np.abs(coefficients)
                
                # Calculate relative importance (normalized)
                relative_importance = importance / np.sum(importance) * 100
                
                importance_df = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Description': [self.feature_descriptions[f] for f in self.feature_names],
                    'Coefficient': coefficients,
                    'Absolute_Importance': importance,
                    'Relative_Importance_%': relative_importance,
                    'Direction': ['Increases Risk' if c > 0 else 'Decreases Risk' for c in coefficients]
                }).sort_values('Absolute_Importance', ascending=False)
                
                return importance_df
            else:
                logger.warning("Model does not have coefficients")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame()
    
    def create_feature_importance_plot(self) -> go.Figure:
        """
        Create enhanced feature importance visualization
        
        Returns:
            Plotly figure object
        """
        importance_df = self.get_feature_importance()
        
        if importance_df.empty:
            return go.Figure().add_annotation(
                text="Model not available for explanation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Create horizontal bar chart with enhanced styling
        fig = go.Figure()
        
        # Color bars based on positive/negative coefficients
        colors = ['#FF6B6B' if coef < 0 else '#4ECDC4' for coef in importance_df['Coefficient']]
        
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Relative_Importance_%'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{imp:.1f}%" for imp in importance_df['Relative_Importance_%']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Relative Importance: %{x:.1f}%<br>' +
                         'Coefficient: %{customdata:.3f}<br>' +
                         '<extra></extra>',
            customdata=importance_df['Coefficient']
        ))
        
        fig.update_layout(
            title={
                'text': 'Feature Importance Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Relative Importance (%)',
            yaxis_title='Features',
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=False)
        
        return fig
    
    def create_coefficient_plot(self) -> go.Figure:
        """
        Create enhanced coefficient visualization showing direction of influence
        
        Returns:
            Plotly figure object
        """
        importance_df = self.get_feature_importance()
        
        if importance_df.empty:
            return go.Figure().add_annotation(
                text="Model not available for explanation",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Sort by coefficient value
        importance_df = importance_df.sort_values('Coefficient')
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        # Color bars based on positive/negative coefficients
        colors = ['#FF6B6B' if coef < 0 else '#4ECDC4' for coef in importance_df['Coefficient']]
        
        fig.add_trace(go.Bar(
            y=importance_df['Feature'],
            x=importance_df['Coefficient'],
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{coef:.3f}" for coef in importance_df['Coefficient']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Coefficient: %{x:.3f}<br>' +
                         'Effect: %{customdata}<br>' +
                         '<extra></extra>',
            customdata=importance_df['Direction']
        ))
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        fig.update_layout(
            title={
                'text': 'Feature Coefficients (Direction of Influence)',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title='Coefficient Value',
            yaxis_title='Features',
            height=500,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        # Add annotations for positive/negative regions
        fig.add_annotation(
            x=max(importance_df['Coefficient']) * 0.7,
            y=len(importance_df) - 0.5,
            text="Increases<br>Diabetes Risk",
            showarrow=False,
            font=dict(color="#4ECDC4", size=10),
            bgcolor="rgba(78, 205, 196, 0.1)",
            bordercolor="#4ECDC4",
            borderwidth=1
        )
        
        fig.add_annotation(
            x=min(importance_df['Coefficient']) * 0.7,
            y=len(importance_df) - 0.5,
            text="Decreases<br>Diabetes Risk",
            showarrow=False,
            font=dict(color="#FF6B6B", size=10),
            bgcolor="rgba(255, 107, 107, 0.1)",
            bordercolor="#FF6B6B",
            borderwidth=1
        )
        
        return fig
    
    def explain_prediction(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Provide detailed explanation for a specific prediction
        
        Args:
            input_data: Dictionary with input features
            
        Returns:
            Dictionary with explanation details
        """
        if self.model is None or self.scaler is None:
            return {'error': 'Model not available'}
        
        try:
            # Prepare input
            input_df = pd.DataFrame([list(input_data.values())], columns=self.feature_names)
            input_scaled = self.scaler.transform(input_df)
            
            # Get prediction and probability
            prediction = self.model.predict(input_scaled)[0]
            probability = self.model.predict_proba(input_scaled)[0]
            
            # Get feature contributions
            coefficients = self.model.coef_[0]
            intercept = self.model.intercept_[0]
            
            # Calculate feature contributions to the prediction
            contributions = coefficients * input_scaled[0]
            
            # Create explanation
            explanation = {
                'prediction': int(prediction),
                'probability_no_diabetes': float(probability[0]),
                'probability_diabetes': float(probability[1]),
                'intercept': float(intercept),
                'feature_contributions': {
                    feature: {
                        'value': float(input_data[feature.lower().replace('bloodpressure', 'bp').replace('skinthickness', 'skin').replace('diabetespedigreefunction', 'pedigree')]),
                        'scaled_value': float(input_scaled[0][i]),
                        'coefficient': float(coefficients[i]),
                        'contribution': float(contributions[i]),
                        'description': self.feature_descriptions[feature]
                    }
                    for i, feature in enumerate(self.feature_names)
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {str(e)}")
            return {'error': str(e)}
    
    def create_prediction_explanation_plot(self, explanation: Dict[str, Any]) -> go.Figure:
        """
        Create visualization for prediction explanation
        
        Args:
            explanation: Explanation dictionary from explain_prediction
            
        Returns:
            Plotly figure object
        """
        if 'error' in explanation:
            return go.Figure().add_annotation(
                text=f"Error: {explanation['error']}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Extract feature contributions
        features = []
        contributions = []
        values = []
        
        for feature, data in explanation['feature_contributions'].items():
            features.append(feature)
            contributions.append(data['contribution'])
            values.append(data['value'])
        
        # Sort by absolute contribution
        sorted_indices = np.argsort(np.abs(contributions))[::-1]
        features = [features[i] for i in sorted_indices]
        contributions = [contributions[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Create waterfall-like chart
        fig = go.Figure()
        
        # Color based on contribution direction
        colors = ['#FF6B6B' if c < 0 else '#4ECDC4' for c in contributions]
        
        fig.add_trace(go.Bar(
            y=features,
            x=contributions,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f"{c:.3f}" for c in contributions],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>' +
                         'Input Value: %{customdata}<br>' +
                         'Contribution: %{x:.3f}<br>' +
                         '<extra></extra>',
            customdata=values
        ))
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.7)
        
        # Add intercept as a separate bar
        fig.add_trace(go.Bar(
            y=['Intercept'],
            x=[explanation['intercept']],
            orientation='h',
            marker=dict(color='gray', opacity=0.7),
            text=[f"{explanation['intercept']:.3f}"],
            textposition='outside',
            name='Model Intercept',
            showlegend=False
        ))
        
        fig.update_layout(
            title={
                'text': f'Prediction Explanation (Probability: {explanation["probability_diabetes"]:.1%})',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis_title='Contribution to Prediction',
            yaxis_title='Features',
            height=600,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=11),
            margin=dict(l=150, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_model_performance_dashboard(self) -> Dict[str, go.Figure]:
        """
        Create comprehensive model performance dashboard
        
        Returns:
            Dictionary of figure objects
        """
        dashboard = {}
        
        try:
            # Load test data for performance evaluation
            df = pd.read_csv("diabetes.csv")
            
            # Preprocess data (same as in training)
            cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
            for col in cols:
                df[col] = df[col].replace(0, df[col].median())
            
            X = df[self.feature_names]
            y = df["Outcome"]
            
            if self.model is not None and self.scaler is not None:
                X_scaled = self.scaler.transform(X)
                y_pred = self.model.predict(X_scaled)
                y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                roc_fig = go.Figure()
                roc_fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'ROC Curve (AUC = {roc_auc:.3f})',
                    line=dict(color='#4ECDC4', width=3)
                ))
                roc_fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    name='Random Classifier',
                    line=dict(color='gray', dash='dash')
                ))
                roc_fig.update_layout(
                    title='ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    showlegend=True
                )
                dashboard['roc_curve'] = roc_fig
                
                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y, y_pred_proba)
                pr_auc = auc(recall, precision)
                
                pr_fig = go.Figure()
                pr_fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    mode='lines',
                    name=f'PR Curve (AUC = {pr_auc:.3f})',
                    line=dict(color='#FF6B6B', width=3)
                ))
                pr_fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision',
                    showlegend=True
                )
                dashboard['pr_curve'] = pr_fig
                
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {str(e)}")
        
        return dashboard
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive model summary
        
        Returns:
            Dictionary with model summary information
        """
        if self.model is None:
            return {'error': 'Model not available'}
        
        try:
            summary = {
                'model_type': type(self.model).__name__,
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names,
                'intercept': float(self.model.intercept_[0]) if hasattr(self.model, 'intercept_') else None,
                'coefficients': self.model.coef_[0].tolist() if hasattr(self.model, 'coef_') else None,
                'solver': getattr(self.model, 'solver', 'unknown'),
                'max_iter': getattr(self.model, 'max_iter', 'unknown'),
                'C': getattr(self.model, 'C', 'unknown')
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return {'error': str(e)}

def display_model_explanation_page():
    """Display comprehensive model explanation page in Streamlit"""
    st.markdown("### 🔍 Model Explanation & Interpretability")
    
    try:
        explainer = ModelExplainer()
        
        if explainer.model is None:
            st.error("❌ Model not available. Please train the model first.")
            return
        
        # Model Summary
        st.markdown("#### 📋 Model Summary")
        summary = explainer.get_model_summary()
        
        if 'error' not in summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", summary['model_type'])
            with col2:
                st.metric("Features", summary['n_features'])
            with col3:
                st.metric("Solver", summary['solver'])
        
        # Feature Importance
        st.markdown("#### 🎯 Feature Importance Analysis")
        
        tab1, tab2 = st.tabs(["📊 Importance", "📈 Coefficients"])
        
        with tab1:
            importance_fig = explainer.create_feature_importance_plot()
            st.plotly_chart(importance_fig, use_container_width=True)
            
            # Feature importance table
            importance_df = explainer.get_feature_importance()
            if not importance_df.empty:
                st.markdown("##### 📋 Detailed Feature Analysis")
                st.dataframe(
                    importance_df[['Feature', 'Description', 'Relative_Importance_%', 'Direction']],
                    use_container_width=True
                )
        
        with tab2:
            coeff_fig = explainer.create_coefficient_plot()
            st.plotly_chart(coeff_fig, use_container_width=True)
            
            st.markdown("""
            **Understanding Coefficients:**
            - **Positive coefficients** (blue): Increase diabetes risk
            - **Negative coefficients** (red): Decrease diabetes risk
            - **Magnitude**: Indicates strength of influence
            """)
        
        # Model Performance Dashboard
        st.markdown("#### 📊 Model Performance Analysis")
        dashboard = explainer.create_model_performance_dashboard()
        
        if dashboard:
            col1, col2 = st.columns(2)
            with col1:
                if 'roc_curve' in dashboard:
                    st.plotly_chart(dashboard['roc_curve'], use_container_width=True)
            with col2:
                if 'pr_curve' in dashboard:
                    st.plotly_chart(dashboard['pr_curve'], use_container_width=True)
        
        # Individual Prediction Explanation
        st.markdown("#### 🔍 Individual Prediction Explanation")
        st.markdown("Enter values to see how each feature contributes to the prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 17, 3)
            glucose = st.number_input("Glucose (mg/dL)", 0, 200, 120)
            bp = st.number_input("Blood Pressure (mmHg)", 0, 122, 70)
            skin = st.number_input("Skin Thickness (mm)", 0, 99, 20)
        
        with col2:
            insulin = st.number_input("Insulin (μU/mL)", 0, 846, 79)
            bmi = st.number_input("BMI", 0.0, 67.1, 32.0)
            pedigree = st.number_input("Pedigree Function", 0.0, 2.5, 0.47)
            age = st.number_input("Age", 21, 90, 33)
        
        if st.button("🔍 Explain This Prediction", type="primary"):
            input_data = {
                'pregnancies': pregnancies,
                'glucose': glucose,
                'bp': bp,
                'skin': skin,
                'insulin': insulin,
                'bmi': bmi,
                'pedigree': pedigree,
                'age': age
            }
            
            explanation = explainer.explain_prediction(input_data)
            
            if 'error' not in explanation:
                # Show prediction result
                prob = explanation['probability_diabetes']
                st.markdown(f"**Prediction:** {'High Risk' if explanation['prediction'] == 1 else 'Low Risk'}")
                st.markdown(f"**Diabetes Probability:** {prob:.1%}")
                
                # Show explanation plot
                exp_fig = explainer.create_prediction_explanation_plot(explanation)
                st.plotly_chart(exp_fig, use_container_width=True)
                
                # Show top contributing factors
                contributions = explanation['feature_contributions']
                sorted_contributions = sorted(
                    contributions.items(),
                    key=lambda x: abs(x[1]['contribution']),
                    reverse=True
                )
                
                st.markdown("##### 🎯 Top Contributing Factors")
                for i, (feature, data) in enumerate(sorted_contributions[:5]):
                    direction = "increases" if data['contribution'] > 0 else "decreases"
                    st.write(f"{i+1}. **{feature}** (value: {data['value']:.1f}) {direction} risk by {abs(data['contribution']):.3f}")
            else:
                st.error(f"Error: {explanation['error']}")
        
    except Exception as e:
        st.error(f"Error in model explanation: {str(e)}")
        logger.error(f"Error in model explanation page: {str(e)}")

if __name__ == "__main__":
    # For testing purposes
    explainer = ModelExplainer()
    if explainer.model is not None:
        importance_df = explainer.get_feature_importance()
        print("Feature Importance:")
        print(importance_df)