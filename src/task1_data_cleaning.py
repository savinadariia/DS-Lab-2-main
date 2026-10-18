"""
Task 1: Comprehensive Data Quality Assessment and Missing Value Analysis

This module implements various data cleaning techniques and missing value handling strategies,
including multiple imputation methods and comprehensive visualization of data quality issues.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
from scipy.stats import f_oneway, chi2_contingency

matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.datasets import fetch_openml
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class DataQualityAnalyzer:
    """
    A comprehensive data quality analyzer that identifies and handles missing values,
    duplicates, and other data quality issues.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.imputed_datasets = {}
        self.missing_indicators = None
        
    def assess_data_quality(self) -> Dict:
        """
        Perform comprehensive data quality assessment.
        
        Returns:
            Dict: Dictionary containing total_rows, total_columns, duplicates, 
                missing_values, missing_percentage, data_types, and memory_usage
        """
        total_rows, total_columns = self.data.shape

        missing_values = self.data.isnull().sum().to_dict()
        missing_percentage = (
            self.data.isnull().mean().mul(100).to_dict()
            if total_rows > 0
            else {col: 0.0 for col in self.data.columns}
        )

        return {
            'total_rows': total_rows,
            'total_columns': total_columns,
            'duplicates': int(self.data.duplicated().sum()),
            'missing_values': missing_values,
            'missing_percentage': missing_percentage,
            'data_types': self.data.dtypes.astype(str).to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / (1024 ** 2)
        }

    def identify_missing_pattern(self, column: str) -> str:
        """
        Attempt to identify the missing data pattern (MCAR, MAR, MNAR).
        
        Parameters:
            column (str): Column name to analyze
            
        Returns:
            str: String indicating likely missing pattern (MCAR, MAR, or MNAR)
        """
        if column not in self.data.columns:
            return "Column not found"

        missing = self.data[column].isnull()

        if not missing.any():
            return "No missing values"

        if missing.all():
            return "Possibly MNAR (Missing Not at Random)"

        # Check whether missingness is associated with other variables.
        for other_col in self.data.columns:
            if other_col == column:
                continue

            observed = self.data.loc[~self.data[other_col].isnull(), [column, other_col]]
            if len(observed) < 10:
                continue

            # Missingness indicator for the target column.
            indicator = observed[column].isnull().astype(int)

            if indicator.nunique() < 2:
                continue

            feature = observed[other_col]

            try:
                if pd.api.types.is_numeric_dtype(feature):
                    groups = [
                        feature[indicator == group].dropna()
                        for group in (0, 1)
                    ]
                    groups = [group for group in groups if len(group) >= 2]

                    if len(groups) == 2:
                        _, p_value = f_oneway(*groups)
                        if p_value < 0.05:
                            return "Likely MAR (Missing at Random)"

                else:
                    contingency = pd.crosstab(feature, indicator)

                    # Require at least two categories and two indicator values.
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        _, p_value, _, _ = chi2_contingency(contingency)
                        if p_value < 0.05:
                            return "Likely MAR (Missing at Random)"

            except (ValueError, TypeError, FloatingPointError):
                continue

        return "Likely MCAR (Missing Completely at Random)"
    
    def create_missing_indicators(self) -> pd.DataFrame:
        """
        Create binary indicator variables for missing values.
        
        Returns:
            pd.DataFrame: DataFrame with binary indicators for missing values
        """
        indicators = self.data.isnull().astype(int)

        # Keep only columns that contain at least one missing value.
        indicators = indicators.loc[:, indicators.any(axis=0)]

        indicators.columns = [
            f"{col}_missing" for col in indicators.columns
        ]

        self.missing_indicators = indicators.copy()
        return indicators
    
    def simple_imputation(self, strategy: str = 'mean') -> pd.DataFrame:
        """
        Perform simple imputation (mean, median, or mode).
        
        Parameters:
            strategy (str): Imputation strategy - 'mean', 'median', or 'most_frequent'
            
        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        if strategy not in ('mean', 'median', 'most_frequent'):
            raise ValueError(
                "strategy must be 'mean', 'median', or 'most_frequent'."
            )

        result = self.data.copy()

        for column in result.columns:
            if not result[column].isnull().any():
                continue

            series = result[column]

            # Mean and median are meaningful for numeric columns only.
            if strategy in ('mean', 'median'):
                if not pd.api.types.is_numeric_dtype(series):
                    continue

                fill_value = (
                    series.mean() if strategy == 'mean'
                    else series.median()
                )

                # An entirely missing column has no statistic to estimate.
                if pd.isna(fill_value):
                    continue

                result[column] = series.fillna(fill_value)

            else:
                # Most frequent value also works for categorical columns.
                modes = series.mode(dropna=True)

                if not modes.empty:
                    result[column] = series.fillna(modes.iloc[0])

        method_name = f"simple_{strategy}"
        self.imputed_datasets[method_name] = result.copy()

        return result
    
    def knn_imputation(self, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Perform KNN imputation on numeric columns.
        
        Parameters:
            n_neighbors (int): Number of neighbors to use for imputation
            
        Returns:
            pd.DataFrame: DataFrame with KNN imputed values
        """
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer.")

        result = self.data.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_data = result[numeric_cols].copy()

        if len(numeric_cols) > 0 and numeric_data.isnull().values.any():
            # KNN cannot estimate a column containing no observed values.
            all_missing = numeric_data.columns[
                numeric_data.isnull().all()
            ].tolist()

            usable_cols = [
                col for col in numeric_cols if col not in all_missing
            ]

            if usable_cols:
                imputer = KNNImputer(
                    n_neighbors=min(n_neighbors, max(1, len(result) - 1))
                )

                imputed_array = imputer.fit_transform(
                    numeric_data[usable_cols]
                )

                for i, col in enumerate(usable_cols):
                    result[col] = imputed_array[:, i]

            # Use median as a fallback for completely empty numeric columns.
            for col in all_missing:
                result[col] = 0.0

        self.imputed_datasets['knn'] = result.copy()
        return result
    
    def iterative_imputation(self, max_iter: int = 10) -> pd.DataFrame:
        """
        Perform multivariate imputation using MICE algorithm.
        
        Parameters:
            max_iter (int): Maximum number of iterations
            
        Returns:
            pd.DataFrame: DataFrame with iteratively imputed values
        """
        if not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer.")

        result = self.data.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        numeric_data = result[numeric_cols].copy()

        if len(numeric_cols) > 0 and numeric_data.isnull().values.any():
            all_missing = numeric_data.columns[
                numeric_data.isnull().all()
            ].tolist()

            usable_cols = [
                col for col in numeric_cols if col not in all_missing
            ]

            if usable_cols:
                imputer = IterativeImputer(
                    max_iter=max_iter,
                    random_state=42,
                    initial_strategy='median'
                )

                imputed_array = imputer.fit_transform(
                    numeric_data[usable_cols]
                )

                for i, col in enumerate(usable_cols):
                    result[col] = imputed_array[:, i]

            # There is no information from which to estimate these columns.
            for col in all_missing:
                result[col] = 0.0

        self.imputed_datasets['iterative'] = result.copy()
        return result

    def compare_imputation_methods(self) -> pd.DataFrame:
        """
        Compare statistical properties across different imputation methods.
        
        Returns:
            DataFrame with comparison statistics
        """
        stats = ['mean', 'std', 'min', 'max']

        # Use numeric columns because describe statistics are numeric.
        numeric_cols = self.original_data.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_cols) == 0:
            return pd.DataFrame()

        original_stats = self.original_data[numeric_cols].describe().reindex(
            stats
        )

        comparison = {'original': original_stats}

        for method_name, imputed_df in self.imputed_datasets.items():
            method_stats = imputed_df[numeric_cols].describe().reindex(stats)
            comparison[method_name] = method_stats

        return pd.concat(comparison, axis=1, names=['method', 'stats'])


    def visualize_missing_data(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create comprehensive visualizations of missing data patterns.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        fig = plt.figure(figsize=figsize)
        
        # 1. Missing data heatmap
        plt.subplot(2, 2, 1)
        msno.matrix(self.data, ax=plt.gca(), sparkline=False)
        plt.title('Missing Data Pattern Heatmap', fontsize=12, fontweight='bold')
        
        # 2. Bar plot of missing percentages
        plt.subplot(2, 2, 2)
        missing_pct = (self.data.isnull().sum() / len(self.data) * 100).sort_values(ascending=False)
        missing_pct = missing_pct[missing_pct > 0]
        if len(missing_pct) > 0:
            missing_pct.plot(kind='bar', color='coral')
            plt.title('Percentage of Missing Values by Column', fontsize=12, fontweight='bold')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            plt.title('Percentage of Missing Values by Column', fontsize=12, fontweight='bold')
        
        # 3. Correlation heatmap of missing indicators
        plt.subplot(2, 2, 3)
        if self.missing_indicators is not None and len(self.missing_indicators.columns) > 0:
            corr_matrix = self.missing_indicators.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Between Missing Value Patterns', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Missing Value Patterns', ha='center', va='center', fontsize=14)
            plt.title('Correlation Between Missing Value Patterns', fontsize=12, fontweight='bold')
        
        # 4. Missing data dendrogram
        plt.subplot(2, 2, 4)
        if self.data.isnull().sum().sum() > 0:
            msno.dendrogram(self.data, ax=plt.gca())
            plt.title('Missing Data Dendrogram', fontsize=12, fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            plt.title('Missing Data Dendrogram', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def visualize_imputation_comparison(self, column: str, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualize distribution comparison for a specific column across imputation methods.
        
        Parameters:
            column (str): Column name to visualize
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        if column not in self.data.select_dtypes(include=[np.number]).columns:
            print(f"Column '{column}' is not numeric or does not exist")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data (with missing values removed for visualization)
        ax = axes[0]
        self.original_data[column].dropna().hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.axvline(self.original_data[column].median(), color='green', linestyle='--', linewidth=2, label='Median')
        ax.set_title(f'Original Data (n={self.original_data[column].notna().sum()})', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Imputed datasets
        methods = list(self.imputed_datasets.keys())
        for idx, method in enumerate(methods[:5], start=1):
            ax = axes[idx]
            imputed_df = self.imputed_datasets[method]
            imputed_df[column].hist(bins=30, ax=ax, color='lightcoral', edgecolor='black', alpha=0.7)
            ax.axvline(imputed_df[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
            ax.axvline(imputed_df[column].median(), color='green', linestyle='--', linewidth=2, label='Median')
            ax.set_title(f'{method.upper()} Imputation', fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplot if less than 5 methods
        if len(methods) < 5:
            axes[5].axis('off')
        
        plt.suptitle(f'Distribution Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_correlation_comparison(self, figsize: Tuple[int, int] = (15, 8)):
        """
        Compare correlation matrices before and after imputation.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Need at least 2 numeric columns for correlation comparison")
            return
        
        n_methods = len(self.imputed_datasets) + 1
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_methods == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Original correlation (pairwise deletion of missing values)
        ax = axes[0]
        corr_original = self.original_data[numeric_cols].corr()
        sns.heatmap(corr_original, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Original Data Correlation', fontweight='bold')
        
        # Imputed correlations
        for idx, (method_name, imputed_df) in enumerate(self.imputed_datasets.items(), start=1):
            if idx < len(axes):
                ax = axes[idx]
                corr_imputed = imputed_df[numeric_cols].corr()
                sns.heatmap(corr_imputed, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
                ax.set_title(f'{method_name.upper()} Imputation', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_methods, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Correlation Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_missing() -> pd.DataFrame:
    """
    Create a sample dataset with various types of missing values for demonstration.
    
    Returns:
        pd.DataFrame: DataFrame with missing values
    """
    np.random.seed(42)
    n_samples = 200
    
    # Create base data
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples),
        'years_employed': np.random.exponential(5, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Introduce MCAR: completely random missing values in age
    mcar_indices = np.random.choice(df.index, size=20, replace=False)
    df.loc[mcar_indices, 'age'] = np.nan
    
    # Introduce MAR: income missing when credit_score is low
    mar_condition = df['credit_score'] < 650
    mar_indices = df[mar_condition].sample(frac=0.4).index
    df.loc[mar_indices, 'income'] = np.nan
    
    # Introduce MNAR: high debt_ratio values are missing (people hide high debt)
    mnar_condition = df['debt_ratio'] > 0.6
    mnar_indices = df[mnar_condition].sample(frac=0.5).index
    df.loc[mnar_indices, 'debt_ratio'] = np.nan
    
    # Add some random missing in years_employed
    random_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[random_indices, 'years_employed'] = np.nan
    
    return df


def main():
    """
    Main function demonstrating the complete Task 1 workflow.
    
    Returns:
        DataQualityAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 1: Comprehensive Data Quality Assessment and Missing Value Analysis")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with missing values...")
    df = create_sample_dataset_with_missing()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = DataQualityAnalyzer(df)
    
    # Assess data quality
    print("1. DATA QUALITY ASSESSMENT")
    print("-" * 80)
    quality_report = analyzer.assess_data_quality()
    print(f"Total Rows: {quality_report['total_rows']}")
    print(f"Total Columns: {quality_report['total_columns']}")
    print(f"Duplicates: {quality_report['duplicates']}")
    print(f"Memory Usage: {quality_report['memory_usage']:.2f} MB")
    print()
    print("Missing Values Summary:")
    for col, count in quality_report['missing_values'].items():
        pct = quality_report['missing_percentage'][col]
        if count > 0:
            print(f"  {col}: {count} ({pct:.1f}%)")
    print()
    
    # Identify missing patterns
    print("2. MISSING DATA PATTERN IDENTIFICATION")
    print("-" * 80)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            pattern = analyzer.identify_missing_pattern(col)
            print(f"{col}: {pattern}")
    print()
    
    # Create missing indicators
    print("3. CREATING MISSING VALUE INDICATORS")
    print("-" * 80)
    indicators = analyzer.create_missing_indicators()
    print(f"Created {len(indicators.columns)} indicator variables")
    print()
    
    # Apply different imputation methods
    print("4. APPLYING IMPUTATION METHODS")
    print("-" * 80)
    
    print("  a) Simple Imputation (Mean)...")
    analyzer.simple_imputation(strategy='mean')
    
    print("  b) Simple Imputation (Median)...")
    analyzer.simple_imputation(strategy='median')
    
    print("  c) KNN Imputation...")
    analyzer.knn_imputation(n_neighbors=5)
    
    print("  d) Iterative Imputation (MICE)...")
    analyzer.iterative_imputation(max_iter=10)
    print()
    
    # Compare methods
    print("5. COMPARING IMPUTATION METHODS")
    print("-" * 80)
    comparison = analyzer.compare_imputation_methods()
    print(comparison)
    print()
    
    # Generate visualizations
    print("6. GENERATING VISUALIZATIONS")
    print("-" * 80)
    print("  Creating missing data visualizations...")
    analyzer.visualize_missing_data()
    plt.savefig('task1_missing_data_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_missing_data_analysis.png")
    
    print("  Creating imputation comparison for 'income'...")
    analyzer.visualize_imputation_comparison('income')
    plt.savefig('task1_imputation_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_imputation_comparison.png")
    
    print("  Creating correlation comparison...")
    analyzer.visualize_correlation_comparison()
    plt.savefig('task1_correlation_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task1_correlation_comparison.png")
    print()
    
    print("=" * 80)
    print("Task 1 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
