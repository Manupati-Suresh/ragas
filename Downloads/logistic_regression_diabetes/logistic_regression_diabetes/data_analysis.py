# Data analysis and visualization module for the Diabetes Prediction App

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Class for comprehensive data analysis and visualization"""
    
    def __init__(self, data_path: str = "diabetes.csv"):
        """
        Initialize the DataAnalyzer
        
        Args:
            data_path: Path to the diabetes dataset
        """
        self.data_path = data_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the diabetes dataset"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Dataset loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def get_dataset_overview(self) -> Dict:
        """
        Get comprehensive dataset overview
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            return {}
        
        overview = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'data_types': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicate_rows': self.df.duplicated().sum(),
            'outcome_distribution': self.df['Outcome'].value_counts().to_dict(),
            'outcome_percentage': (self.df['Outcome'].value_counts(normalize=True) * 100).to_dict()
        }
        
        return overview
    
    def get_descriptive_statistics(self) -> pd.DataFrame:
        """
        Get descriptive statistics for all numeric columns
        
        Returns:
            DataFrame with descriptive statistics
        """
        if self.df is None:
            return pd.DataFrame()
        
        return self.df.describe()
    
    def create_correlation_heatmap(self) -> go.Figure:
        """
        Create correlation heatmap
        
        Returns:
            Plotly figure object
        """
        if self.df is None:
            return go.Figure()
        
        corr_matrix = self.df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Feature Correlation Heatmap',
            xaxis_title='Features',
            yaxis_title='Features',
            width=800,
            height=600
        )
        
        return fig
    
    def create_distribution_plots(self) -> go.Figure:
        """
        Create distribution plots for all features
        
        Returns:
            Plotly figure object with subplots
        """
        if self.df is None:
            return go.Figure()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Outcome']
        
        rows = (len(numeric_cols) + 3) // 4
        fig = make_subplots(
            rows=rows, 
            cols=4,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // 4 + 1
            col_pos = i % 4 + 1
            
            # Create histogram
            fig.add_trace(
                go.Histogram(
                    x=self.df[col],
                    name=col,
                    showlegend=False,
                    opacity=0.7
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title='Feature Distributions',
            height=300 * rows,
            showlegend=False
        )
        
        return fig
    
    def create_outcome_comparison_plots(self) -> go.Figure:
        """
        Create box plots comparing features by outcome
        
        Returns:
            Plotly figure object
        """
        if self.df is None:
            return go.Figure()
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Outcome']
        
        rows = (len(numeric_cols) + 3) // 4
        fig = make_subplots(
            rows=rows, 
            cols=4,
            subplot_titles=numeric_cols,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for i, col in enumerate(numeric_cols):
            row = i // 4 + 1
            col_pos = i % 4 + 1
            
            # Box plot for non-diabetic
            fig.add_trace(
                go.Box(
                    y=self.df[self.df['Outcome'] == 0][col],
                    name='Non-Diabetic',
                    showlegend=(i == 0),
                    marker_color='lightblue'
                ),
                row=row, col=col_pos
            )
            
            # Box plot for diabetic
            fig.add_trace(
                go.Box(
                    y=self.df[self.df['Outcome'] == 1][col],
                    name='Diabetic',
                    showlegend=(i == 0),
                    marker_color='lightcoral'
                ),
                row=row, col=col_pos
            )
        
        fig.update_layout(
            title='Feature Comparison by Diabetes Outcome',
            height=300 * rows
        )
        
        return fig
    
    def create_age_analysis(self) -> go.Figure:
        """
        Create age-based analysis
        
        Returns:
            Plotly figure object
        """
        if self.df is None:
            return go.Figure()
        
        # Create age groups
        age_bins = [20, 30, 40, 50, 60, 70, 80]
        age_labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
        
        df_copy = self.df.copy()
        df_copy['AgeGroup'] = pd.cut(df_copy['Age'], bins=age_bins, labels=age_labels, right=False)
        
        # Calculate diabetes rate by age group
        age_diabetes_rate = df_copy.groupby('AgeGroup')['Outcome'].agg(['count', 'sum', 'mean']).reset_index()
        age_diabetes_rate['diabetes_rate'] = age_diabetes_rate['mean'] * 100
        
        fig = go.Figure()
        
        # Bar chart for diabetes rate
        fig.add_trace(go.Bar(
            x=age_diabetes_rate['AgeGroup'],
            y=age_diabetes_rate['diabetes_rate'],
            name='Diabetes Rate (%)',
            marker_color='lightcoral'
        ))
        
        fig.update_layout(
            title='Diabetes Rate by Age Group',
            xaxis_title='Age Group',
            yaxis_title='Diabetes Rate (%)',
            showlegend=True
        )
        
        return fig
    
    def create_bmi_analysis(self) -> go.Figure:
        """
        Create BMI-based analysis
        
        Returns:
            Plotly figure object
        """
        if self.df is None:
            return go.Figure()
        
        # Create BMI categories
        df_copy = self.df.copy()
        df_copy['BMI_Category'] = pd.cut(
            df_copy['BMI'],
            bins=[0, 18.5, 25, 30, float('inf')],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Calculate diabetes rate by BMI category
        bmi_diabetes_rate = df_copy.groupby('BMI_Category')['Outcome'].agg(['count', 'sum', 'mean']).reset_index()
        bmi_diabetes_rate['diabetes_rate'] = bmi_diabetes_rate['mean'] * 100
        
        fig = go.Figure()
        
        # Bar chart for diabetes rate
        fig.add_trace(go.Bar(
            x=bmi_diabetes_rate['BMI_Category'],
            y=bmi_diabetes_rate['diabetes_rate'],
            name='Diabetes Rate (%)',
            marker_color='lightgreen'
        ))
        
        fig.update_layout(
            title='Diabetes Rate by BMI Category',
            xaxis_title='BMI Category',
            yaxis_title='Diabetes Rate (%)',
            showlegend=True
        )
        
        return fig
    
    def create_glucose_analysis(self) -> go.Figure:
        """
        Create glucose-based analysis
        
        Returns:
            Plotly figure object
        """
        if self.df is None:
            return go.Figure()
        
        # Create glucose categories
        df_copy = self.df.copy()
        df_copy['Glucose_Category'] = pd.cut(
            df_copy['Glucose'],
            bins=[0, 100, 126, float('inf')],
            labels=['Normal', 'Pre-diabetic', 'Diabetic Range']
        )
        
        # Calculate diabetes rate by glucose category
        glucose_diabetes_rate = df_copy.groupby('Glucose_Category')['Outcome'].agg(['count', 'sum', 'mean']).reset_index()
        glucose_diabetes_rate['diabetes_rate'] = glucose_diabetes_rate['mean'] * 100
        
        fig = go.Figure()
        
        # Bar chart for diabetes rate
        fig.add_trace(go.Bar(
            x=glucose_diabetes_rate['Glucose_Category'],
            y=glucose_diabetes_rate['diabetes_rate'],
            name='Diabetes Rate (%)',
            marker_color='gold'
        ))
        
        fig.update_layout(
            title='Diabetes Rate by Glucose Category',
            xaxis_title='Glucose Category',
            yaxis_title='Diabetes Rate (%)',
            showlegend=True
        )
        
        return fig
    
    def create_comprehensive_dashboard(self) -> Dict[str, go.Figure]:
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Returns:
            Dictionary of figure objects
        """
        dashboard = {}
        
        try:
            dashboard['correlation_heatmap'] = self.create_correlation_heatmap()
            dashboard['distribution_plots'] = self.create_distribution_plots()
            dashboard['outcome_comparison'] = self.create_outcome_comparison_plots()
            dashboard['age_analysis'] = self.create_age_analysis()
            dashboard['bmi_analysis'] = self.create_bmi_analysis()
            dashboard['glucose_analysis'] = self.create_glucose_analysis()
            
            logger.info("Comprehensive dashboard created successfully")
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
        
        return dashboard
    
    def get_feature_insights(self) -> Dict[str, str]:
        """
        Get insights about each feature
        
        Returns:
            Dictionary with feature insights
        """
        if self.df is None:
            return {}
        
        insights = {}
        
        # Glucose insights
        glucose_diabetic = self.df[self.df['Outcome'] == 1]['Glucose'].mean()
        glucose_non_diabetic = self.df[self.df['Outcome'] == 0]['Glucose'].mean()
        insights['Glucose'] = f"Average glucose: {glucose_diabetic:.1f} mg/dL (diabetic) vs {glucose_non_diabetic:.1f} mg/dL (non-diabetic)"
        
        # BMI insights
        bmi_diabetic = self.df[self.df['Outcome'] == 1]['BMI'].mean()
        bmi_non_diabetic = self.df[self.df['Outcome'] == 0]['BMI'].mean()
        insights['BMI'] = f"Average BMI: {bmi_diabetic:.1f} (diabetic) vs {bmi_non_diabetic:.1f} (non-diabetic)"
        
        # Age insights
        age_diabetic = self.df[self.df['Outcome'] == 1]['Age'].mean()
        age_non_diabetic = self.df[self.df['Outcome'] == 0]['Age'].mean()
        insights['Age'] = f"Average age: {age_diabetic:.1f} years (diabetic) vs {age_non_diabetic:.1f} years (non-diabetic)"
        
        # Pregnancies insights
        preg_diabetic = self.df[self.df['Outcome'] == 1]['Pregnancies'].mean()
        preg_non_diabetic = self.df[self.df['Outcome'] == 0]['Pregnancies'].mean()
        insights['Pregnancies'] = f"Average pregnancies: {preg_diabetic:.1f} (diabetic) vs {preg_non_diabetic:.1f} (non-diabetic)"
        
        return insights

def display_data_analysis_page():
    """Display the data analysis page in Streamlit"""
    st.markdown("### 📊 Dataset Analysis & Insights")
    
    try:
        analyzer = DataAnalyzer()
        
        # Dataset overview
        st.markdown("#### 📋 Dataset Overview")
        overview = analyzer.get_dataset_overview()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", overview['shape'][0])
        with col2:
            st.metric("Features", overview['shape'][1] - 1)  # Excluding outcome
        with col3:
            st.metric("Diabetic Cases", overview['outcome_distribution'].get(1, 0))
        with col4:
            st.metric("Diabetes Rate", f"{overview['outcome_percentage'].get(1, 0):.1f}%")
        
        # Descriptive statistics
        st.markdown("#### 📈 Descriptive Statistics")
        st.dataframe(analyzer.get_descriptive_statistics())
        
        # Feature insights
        st.markdown("#### 💡 Key Insights")
        insights = analyzer.get_feature_insights()
        for feature, insight in insights.items():
            st.write(f"**{feature}:** {insight}")
        
        # Visualizations
        dashboard = analyzer.create_comprehensive_dashboard()
        
        # Correlation heatmap
        st.markdown("#### 🔗 Feature Correlations")
        st.plotly_chart(dashboard['correlation_heatmap'], use_container_width=True)
        
        # Distribution plots
        st.markdown("#### 📊 Feature Distributions")
        st.plotly_chart(dashboard['distribution_plots'], use_container_width=True)
        
        # Outcome comparison
        st.markdown("#### 📈 Feature Comparison by Outcome")
        st.plotly_chart(dashboard['outcome_comparison'], use_container_width=True)
        
        # Age analysis
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(dashboard['age_analysis'], use_container_width=True)
        with col2:
            st.plotly_chart(dashboard['bmi_analysis'], use_container_width=True)
        
        # Glucose analysis
        st.plotly_chart(dashboard['glucose_analysis'], use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in data analysis: {str(e)}")
        logger.error(f"Error in data analysis page: {str(e)}")

if __name__ == "__main__":
    # For testing purposes
    analyzer = DataAnalyzer()
    overview = analyzer.get_dataset_overview()
    print("Dataset Overview:", overview)