"""
Task 2: Outlier Detection and Treatment with Advanced Visualization

This module implements various outlier detection and treatment techniques,
including statistical methods, machine learning-based detection, and 
comprehensive visualization of outliers and their impact.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
    """
    A comprehensive outlier analyzer that detects and treats outliers using
    multiple methods and provides detailed visualizations.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the analyzer with a dataset.
        
        Parameters:
            data (pd.DataFrame): Input DataFrame to analyze
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.outlier_masks = {}
        self.treated_datasets = {}

    def _numeric_data(self):
        return self.data.select_dtypes(include=[np.number])

    def _get_mask(self, method='iqr'):
        """Return a previously calculated or newly calculated outlier mask."""
        methods = {
            'zscore': self.detect_outliers_zscore,
            'iqr': self.detect_outliers_iqr,
            'isolation_forest': self.detect_outliers_isolation_forest,
            'lof': self.detect_outliers_lof
        }
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Choose from {list(methods)}")
        return methods[method]()
        
    def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using Z-score method.
        
        Parameters:
            threshold (float): Z-score threshold for outlier detection
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if threshold <= 0:
            raise ValueError("threshold must be positive.")

        numeric = self._numeric_data()
        std = numeric.std(ddof=0).replace(0, np.nan)

        z_scores = (numeric - numeric.mean()) / std
        mask = z_scores.abs().gt(threshold).fillna(False)

        self.outlier_masks['zscore'] = mask
        return mask
    
    def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
            multiplier (float): IQR multiplier for boundary calculation
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if multiplier < 0:
            raise ValueError("multiplier cannot be negative.")

        numeric = self._numeric_data()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        mask = numeric.lt(lower, axis='columns') | numeric.gt(
            upper, axis='columns'
        )
        mask = mask.fillna(False)

        self.outlier_masks['iqr'] = mask
        return mask

    def detect_outliers_isolation_forest(self, contamination: float = 0.1,
                                        random_state: int = 42) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
            contamination (float): Expected proportion of outliers in the dataset
            random_state (int): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5].")

        numeric = self._numeric_data()
        mask = pd.DataFrame(
            False, index=numeric.index, columns=numeric.columns
        )

        if numeric.shape[1] == 0:
            self.outlier_masks['isolation_forest'] = mask
            return mask

        # Model cannot accept NaN or infinite values.
        values = numeric.replace([np.inf, -np.inf], np.nan)
        valid_rows = values.notna().any(axis=1)

        if valid_rows.sum() < 2:
            self.outlier_masks['isolation_forest'] = mask
            return mask

        values = values.loc[valid_rows]
        values = values.fillna(values.median()).fillna(0)

        if len(values) < 2:
            self.outlier_masks['isolation_forest'] = mask
            return mask

        model = make_pipeline(
            StandardScaler(),
            IsolationForest(
                contamination=contamination,
                random_state=random_state
            )
        )
        predictions = model.fit_predict(values)
        outlier_rows = values.index[predictions == -1]

        # Mark all numeric cells in an anomalous row.
        mask.loc[outlier_rows, :] = True

        self.outlier_masks['isolation_forest'] = mask
        return mask
    
    def detect_outliers_lof(self, n_neighbors: int = 20, 
                           contamination: float = 0.1) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF) algorithm.
        
        Parameters:
            n_neighbors (int): Number of neighbors to consider
            contamination (float): Expected proportion of outliers
            
        Returns:
            pd.DataFrame: Boolean mask indicating outliers
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive.")
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5].")

        numeric = self._numeric_data()
        mask = pd.DataFrame(
            False, index=numeric.index, columns=numeric.columns
        )

        values = numeric.replace([np.inf, -np.inf], np.nan)
        valid_rows = values.notna().any(axis=1)
        values = values.loc[valid_rows]

        if len(values) < 3 or numeric.shape[1] == 0:
            self.outlier_masks['lof'] = mask
            return mask

        values = values.fillna(values.median()).fillna(0)
        values = pd.DataFrame(
            StandardScaler().fit_transform(values),
            index=values.index,
            columns=values.columns
        )

        # LOF requires n_neighbors < number of samples.
        k = min(n_neighbors, len(values) - 1)

        model = LocalOutlierFactor(
            n_neighbors=k,
            contamination=contamination
        )
        predictions = model.fit_predict(values)
        outlier_rows = values.index[predictions == -1]

        mask.loc[outlier_rows, :] = True

        self.outlier_masks['lof'] = mask
        return mask
    
    def get_outlier_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of outliers detected by different methods.
        
        Returns:
            DataFrame with outlier counts for each method
        """
        if not self.outlier_masks:
            self.detect_outliers_zscore()
            self.detect_outliers_iqr()
            self.detect_outliers_isolation_forest()
            self.detect_outliers_lof()

        numeric = self._numeric_data()
        records = []

        for method, mask in self.outlier_masks.items():
            aligned = mask.reindex(
                index=numeric.index,
                columns=numeric.columns,
                fill_value=False
            )

            row_count = int(aligned.any(axis=1).sum())
            cell_count = int(aligned.sum().sum())
            percentage = (
                row_count / len(numeric) * 100
                if len(numeric) > 0
                else 0.0
            )

            records.append({
                'method': method,
                'outlier_cells': cell_count,
                'outlier_rows': row_count,
                'percentage_of_rows': percentage
            })

        if not records:
            return pd.DataFrame(
                index=['outlier_cells', 'outlier_rows', 'percentage_of_rows']
            )

        summary = pd.DataFrame(records).set_index('method')

        return summary.T
    
    def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from the dataset.
        
        Parameters:
            method (str): Detection method to use
            
        Returns:
            pd.DataFrame: Dataset with outliers removed
        """
        mask = self._get_mask(method)
        row_mask = mask.any(axis=1)

        # Preserve rows with no flagged outlier in any numeric column.
        result = self.data.loc[~row_mask.reindex(
            self.data.index, fill_value=False
        )].copy()

        self.treated_datasets[f'removed_{method}'] = result.copy()
        return result
    
    def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
        """
        Cap outliers using winsorization.
        
        Parameters:
            method (str): Detection method to use
            multiplier (float): IQR multiplier for bounds
            
        Returns:
            pd.DataFrame: Dataset with capped outliers
        """
        result = self.data.copy()
        numeric = self._numeric_data()

        if method == 'iqr':
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            spread = q3 - q1
            lower = q1 - multiplier * spread
            upper = q3 + multiplier * spread

        elif method == 'zscore':
            means = numeric.mean()
            stds = numeric.std(ddof=0)
            lower = means - multiplier * stds
            upper = means + multiplier * stds

        else:
            raise ValueError("Capping supports method='iqr' or 'zscore'.")

        for col in numeric.columns:
            if pd.notna(lower[col]) and pd.notna(upper[col]):
                result[col] = result[col].clip(
                    lower=lower[col], upper=upper[col]
                )

        self.treated_datasets['capped'] = result.copy()
        return result
    
    def transform_outliers_log(self) -> pd.DataFrame:
        """
        Transform data using log transformation.
        
        Returns:
            pd.DataFrame: Dataset with log-transformed values
        """
        result = self.data.copy()
        numeric = self._numeric_data()

        for col in numeric.columns:
            valid = numeric[col].dropna()

            if len(valid) == 0:
                continue

            # log1p is real-valued for values greater than -1.
            if (valid > -1).all():
                result[col] = np.log1p(result[col])
            else:
                # Shift the column to make all finite values nonnegative.
                min_value = valid.min()
                shift = max(0.0, -min_value)
                result[col] = np.log1p(result[col] + shift)

        self.treated_datasets['log_transform'] = result.copy()
        return result
    
    def transform_outliers_boxcox(self) -> pd.DataFrame:
        """
        Transform data using Box-Cox/Yeo-Johnson transformation.
        
        Returns:
            pd.DataFrame: Dataset with power-transformed values
        """
        result = self.data.copy()
        numeric = self._numeric_data()

        # Yeo-Johnson supports both positive and negative values.
        # Fit each column independently to handle constant columns safely.
        for col in numeric.columns:
            series = numeric[col]
            valid = series.dropna()

            if len(valid) < 2 or valid.nunique() < 2:
                continue

            transformer = PowerTransformer(method='yeo-johnson')
            result.loc[valid.index, col] = transformer.fit_transform(
                valid.to_numpy().reshape(-1, 1)
            ).ravel()

        self.treated_datasets['boxcox_transform'] = result.copy()
        return result
    
    def evaluate_model_performance(self, target_col: str, 
                                   dataset_name: str = 'original') -> Dict[str, float]:
        """
        Evaluate model performance with and without outliers.
        
        Args:
            target_col: Name of target column
            dataset_name: Dataset to evaluate ('original' or treated dataset name)
            
        Returns:
            Dictionary with performance metrics
        """
        if dataset_name == 'original':
            dataset = self.original_data.copy()
        elif dataset_name in self.treated_datasets:
            dataset = self.treated_datasets[dataset_name].copy()
        else:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: "
                f"{['original'] + list(self.treated_datasets.keys())}"
            )

        if target_col not in dataset.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        numeric = dataset.select_dtypes(include=[np.number]).copy()

        if target_col not in numeric.columns:
            raise ValueError("Target column must be numeric.")

        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.dropna(subset=[target_col])

        feature_cols = [c for c in numeric.columns if c != target_col]

        if not feature_cols:
            raise ValueError("At least one numeric feature is required.")

        X = numeric[feature_cols]
        y = numeric[target_col]

        # Drop entirely empty features; impute remaining missing values
        # within each CV fold to avoid leakage.
        X = X.dropna(axis=1, how='all')

        if X.shape[1] == 0:
            raise ValueError("No usable numeric feature columns.")

        n_splits = min(5, len(numeric))

        if n_splits < 2:
            raise ValueError("At least two usable rows are required.")

        model = make_pipeline(
            SimpleImputer(strategy='median'),
            LinearRegression()
        )

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model, X, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=42),
            scoring="r2"
        )

        return {
            "r2_score": float(np.mean(cv_scores)),
            "r2_std": float(np.std(cv_scores)),
            "r2_scores": cv_scores.tolist(),
            "mean_cv_score": float(np.mean(cv_scores))
        }

    def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create box plots showing outliers.
        
        Parameters:
            figsize (Tuple[int, int]): Figure size for plots
            
        Returns:
            matplotlib.figure.Figure: The generated figure object
        """
        numeric = self._numeric_data()

        if numeric.empty:
            print("No numeric columns to visualize.")
            return

        cols = numeric.columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = np.asarray(axes).flatten()

        for i, col in enumerate(cols):
            sns.boxplot(y=numeric[col], ax=axes[i], color='skyblue')
            axes[i].set_title(f'Boxplot: {col}', fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3)

        for i in range(len(cols), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Outlier Detection: Boxplots', fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create violin plots showing distribution with outliers.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]

        numeric_cols = self._numeric_data().columns[:6]

        if len(numeric_cols) == 0:
            print("No numeric columns to visualize.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = np.asarray(axes).flatten()

        for i, col in enumerate(numeric_cols):
            sns.violinplot(
                y=self.data[col], ax=axes[i], color='lightcoral',
                inner='quartile'
            )
            axes[i].set_title(f'Violin Plot: {col}', fontweight='bold')
            axes[i].grid(axis='y', alpha=0.3)

        for i in range(len(numeric_cols), len(axes)):
            axes[i].axis('off')

        plt.suptitle('Outlier Distribution: Violin Plots', fontweight='bold')
        plt.tight_layout()
        return fig

    def visualize_outliers_scatter(
            self,
            x_col: str,
            y_col: str,
            figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Create scatter plots showing outliers detected by different methods.

        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            figsize: Figure size for plots
        """

        # Validate columns before accessing the DataFrame
        for col in (x_col, y_col):
            if col not in self.data.columns:
                print(f"Column '{col}' not found.")
                return None

            if not pd.api.types.is_numeric_dtype(self.data[col]):
                print(f"Column '{col}' must be numeric.")
                return None

        methods = ['zscore', 'iqr', 'isolation_forest', 'lof']
        fig, axes = plt.subplots(1, 4, figsize=figsize)

        for ax, method in zip(axes, methods):
            mask = self._get_mask(method)
            aligned = mask.reindex(
                self.data.index, fill_value=False
            )

            if x_col in aligned.columns and y_col in aligned.columns:
                outliers = aligned[x_col] | aligned[y_col]
            else:
                outliers = pd.Series(False, index=self.data.index)

            # Separate normal observations and outliers
            ax.scatter(
                self.data.loc[~outliers, x_col],
                self.data.loc[~outliers, y_col],
                alpha=0.6,
                label='Normal'
            )

            ax.scatter(
                self.data.loc[outliers, x_col],
                self.data.loc[outliers, y_col],
                color='red',
                marker='x',
                label='Outlier'
            )

            ax.set_title(method.replace('_', ' ').title())
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.legend()
            ax.grid(alpha=0.2)

        plt.tight_layout()
        return fig
    
    def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str, 
                             method: str = 'isolation_forest',
                             figsize: Tuple[int, int] = (12, 8)):
        """
        Create three-dimensional scatter plot showing multivariate outliers.
        
        Args:
            x_col, y_col, z_col: Columns for three-dimensional axes
            method: Detection method to use
            figsize: Figure size for plot
        """
        if method not in self.outlier_masks:
            print(f"Outlier detection results for method '{method}' not found.")
            return

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        cols = [x_col, y_col, z_col]

        for col in cols:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found.")
            if not pd.api.types.is_numeric_dtype(self.data[col]):
                raise ValueError(f"Column '{col}' must be numeric.")

        mask = self._get_mask(method)
        row_mask = mask.reindex(
            self.data.index, fill_value=False
        ).any(axis=1)

        valid = self.data[cols].replace(
            [np.inf, -np.inf], np.nan
        ).notna().all(axis=1)

        normal = self.data.loc[valid & ~row_mask, cols]
        outliers = self.data.loc[valid & row_mask, cols]


        fig = plt.figure(figsize=(float(figsize[0]), float(figsize[1])))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            normal[x_col], normal[y_col], normal[z_col],
            label='Normal', alpha=0.6
        )
        ax.scatter(
            outliers[x_col], outliers[y_col], outliers[z_col],
            color='red', marker='x', s=60, label='Outlier'
        )

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f'3D Outlier Detection: {method}')
        ax.legend()

        plt.tight_layout()
        return fig
    
    def visualize_treatment_comparison(self, column: str, 
                                      figsize: Tuple[int, int] = (15, 10)):
        """
        Compare distributions before and after different treatments.
        
        Args:
            column: Column to visualize
            figsize: Figure size for plots
        """
        if column not in self.data.columns:
            print(f"Column '{column}' not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Original data
        ax = axes[0]
        self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Original Data', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Treated datasets
        treatment_names = list(self.treated_datasets.keys())[:5]
        for idx, name in enumerate(treatment_names, start=1):
            ax = axes[idx]
            treated_data = self.treated_datasets[name]
            if column in treated_data.columns:
                treated_data[column].hist(bins=30, ax=ax, color='lightcoral', 
                                        edgecolor='black', alpha=0.7)
                ax.axvline(treated_data[column].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Mean')
                ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(treatment_names) + 1, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig
    
    def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Create Q-Q plots to assess normality before and after treatment.
        
        Args:
            figsize: Figure size for plots
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            print("No numeric columns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            
            stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
    """
    Create a sample dataset with outliers for demonstration.
    
    Returns:
        pd.DataFrame: Dataset with outliers
    """
    np.random.seed(42)
    n_samples = 300
    
    # Create base data
    data = {
        'feature1': np.random.normal(100, 15, n_samples),
        'feature2': np.random.exponential(50, n_samples),
        'feature3': np.random.normal(500, 100, n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.normal(200, 30, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add synthetic target relationship
    df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)
    
    # Inject outliers
    # Extreme high values in feature1
    outlier_indices = np.random.choice(df.index, size=15, replace=False)
    df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))
    
    # Extreme low values in feature2
    outlier_indices = np.random.choice(df.index, size=10, replace=False)
    df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))
    
    # Random extreme values in feature3
    outlier_indices = np.random.choice(df.index, size=20, replace=False)
    # Split outliers into low and high
    low_outliers = outlier_indices[:len(outlier_indices)//2]
    high_outliers = outlier_indices[len(outlier_indices)//2:]
    df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
    df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))
    
    return df


def main():
    """
    Main function demonstrating the complete Task 2 workflow.
    
    Returns:
        OutlierAnalyzer: Configured analyzer object
    """
    print("=" * 80)
    print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
    print("=" * 80)
    print()
    
    # Create sample dataset
    print("Creating sample dataset with outliers...")
    df = create_sample_dataset_with_outliers()
    print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print()
    
    # Initialize analyzer
    analyzer = OutlierAnalyzer(df)
    
    # Detect outliers using different methods
    print("1. OUTLIER DETECTION")
    print("-" * 80)
    
    print("  a) Z-Score Detection...")
    zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
    print(f"     Outliers detected: {zscore_outliers.sum().sum()}")
    
    print("  b) IQR Detection...")
    iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
    print(f"     Outliers detected: {iqr_outliers.sum().sum()}")
    
    print("  c) Isolation Forest Detection...")
    iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
    print(f"     Outliers detected: {iso_outliers.sum().sum()}")
    
    print("  d) Local Outlier Factor Detection...")
    lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
    print(f"     Outliers detected: {lof_outliers.sum().sum()}")
    print()
    
    # Get outlier summary
    print("2. OUTLIER SUMMARY")
    print("-" * 80)
    summary = analyzer.get_outlier_summary()
    print(summary)
    print()
    
    # Apply different treatment strategies
    print("3. OUTLIER TREATMENT")
    print("-" * 80)
    
    print("  a) Removing outliers (IQR method)...")
    removed_data = analyzer.remove_outliers(method='iqr')
    print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")
    
    print("  b) Capping outliers (Winsorization)...")
    capped_data = analyzer.cap_outliers(multiplier=1.5)
    print(f"     Data capped at IQR bounds")
    
    print("  c) Log transformation...")
    log_data = analyzer.transform_outliers_log()
    print(f"     Log transformation applied")
    
    print("  d) Box-Cox transformation...")
    boxcox_data = analyzer.transform_outliers_boxcox()
    print(f"     Box-Cox/Yeo-Johnson transformation applied")
    print()
    
    # Evaluate model performance
    print("4. MODEL PERFORMANCE COMPARISON")
    print("-" * 80)
    
    datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
    for dataset_name in datasets_to_evaluate:
        try:
            performance = analyzer.evaluate_model_performance('target', dataset_name)
            print(f"  {dataset_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
        except Exception as e:
            print(f"  {dataset_name}: Could not evaluate - {str(e)}")
    print()
    
    # Generate visualizations
    print("5. GENERATING VISUALIZATIONS")
    print("-" * 80)
    
    print("  Creating box plots...")
    analyzer.visualize_outliers_boxplot()
    plt.savefig('task2_boxplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_boxplots.png")
    
    print("  Creating violin plots...")
    analyzer.visualize_outliers_violin()
    plt.savefig('task2_violin_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_violin_plots.png")
    
    print("  Creating scatter plots with outlier detection...")
    analyzer.visualize_outliers_scatter('feature1', 'feature2')
    plt.savefig('task2_scatter_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_scatter_outliers.png")
    
    print("  Creating 3D outlier visualization...")
    analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
    plt.savefig('task2_3d_outliers.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_3d_outliers.png")
    
    print("  Creating treatment comparison plots...")
    analyzer.visualize_treatment_comparison('feature1')
    plt.savefig('task2_treatment_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_treatment_comparison.png")
    
    print("  Creating Q-Q plots...")
    analyzer.visualize_qq_plots()
    plt.savefig('task2_qq_plots.png', dpi=300, bbox_inches='tight')
    print("  Saved: task2_qq_plots.png")
    print()
    
    print("=" * 80)
    print("Task 2 completed successfully!")
    print("=" * 80)
    
    return analyzer


if __name__ == "__main__":
    analyzer = main()
    plt.show()
