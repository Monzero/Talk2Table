import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional
import re
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate


class DataFrameAnalyzer:
    """
    Extracts basic information and statistics from a DataFrame.
    This class focuses solely on objective metrics and facts about the data.
    """
    
    def __init__(self, df=None, verbose=True):
        """
        Initialize the DataFrame analyzer.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze
            verbose (bool): Whether to print progress information
        """
        self.df = df.copy() if df is not None else None
        self.verbose = verbose
        self.total_rows = len(self.df) if df is not None else 0
        self.basic_info = {}
    
    def analyze(self, df=None) -> Dict[str, Any]:
        """
        Extract basic information and statistics from the DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to analyze
            
        Returns:
            Dict[str, Any]: Basic information about the DataFrame
        """
        if df is not None:
            self.df = df.copy()
        
        if self.df is None:
            raise ValueError("No DataFrame provided for analysis")
        
        if self.verbose:
            print(f"Analyzing DataFrame with {len(self.df)} rows and {len(self.df.columns)} columns")
        
        self._extract_basic_info()
        self._analyze_columns()
        self._check_duplicates()
        self._check_correlations()
        
        return self.basic_info
    
    def _extract_basic_info(self):
        """
        Extract basic information about the DataFrame.
        """
        try:
            # Get basic dataframe info
            self.basic_info = {
                "row_count": len(self.df),
                "column_count": len(self.df.columns),
                "column_names": list(self.df.columns),
                "column_types": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            }
            
            # Calculate missing values per column
            missing_values = {}
            missing_pct = {}
            for col in self.df.columns:
                missing_count = self.df[col].isna().sum()
                missing_values[col] = int(missing_count)
                missing_pct[col] = float(missing_count / len(self.df) * 100)
            
            self.basic_info["missing_values"] = missing_values
            self.basic_info["missing_pct"] = missing_pct
            
            # Calculate total missing values
            total_missing = sum(missing_values.values())
            self.basic_info["total_missing_values"] = int(total_missing)
            self.basic_info["total_missing_pct"] = float(total_missing / (len(self.df) * len(self.df.columns)) * 100)
            
            # Identify columns with high missing values (>20%)
            high_missing = [col for col, pct in missing_pct.items() if pct > 20]
            if high_missing:
                self.basic_info["high_missing_columns"] = high_missing
            
            # Calculate memory usage
            memory_usage = self.df.memory_usage(deep=True).sum()
            if memory_usage < 1024:
                memory_str = f"{memory_usage} bytes"
            elif memory_usage < 1024 * 1024:
                memory_str = f"{memory_usage / 1024:.2f} KB"
            else:
                memory_str = f"{memory_usage / (1024 * 1024):.2f} MB"
            
            self.basic_info["memory_usage"] = memory_str
            
        except Exception as e:
            if self.verbose:
                print(f"Error extracting basic info: {str(e)}")
            # Initialize with minimal information
            self.basic_info = {
                "row_count": len(self.df) if self.df is not None else 0,
                "column_count": len(self.df.columns) if self.df is not None else 0,
                "column_names": list(self.df.columns) if self.df is not None else [],
                "error": str(e)
            }
    
    def _analyze_columns(self):
        """
        Analyze each column in the DataFrame.
        """
        if "column_types" not in self.basic_info:
            return
            
        column_details = {}
        
        for col in self.df.columns:
            try:
                details = self._analyze_single_column(col)
                if details:
                    column_details[col] = details
            except Exception as e:
                if self.verbose:
                    print(f"Error analyzing column {col}: {str(e)}")
                column_details[col] = {
                    "error": str(e)
                }
        
        self.basic_info["column_details"] = column_details
        
        # Detect date columns
        date_cols = [col for col, details in column_details.items() 
                    if details.get("seems_like_date", False)]
        
        if date_cols:
            self.basic_info["date_columns"] = date_cols
            for col in date_cols:
                try:
                    # Use errors='coerce' to handle unparseable values silently
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    valid_dates = dates.dropna()
                    if len(valid_dates) > 0:
                        self.basic_info[f"{col}_min_date"] = valid_dates.min().strftime('%Y-%m-%d')
                        self.basic_info[f"{col}_max_date"] = valid_dates.max().strftime('%Y-%m-%d')
                except Exception as e:
                    if self.verbose:
                        print(f"Error extracting date range for {col}: {str(e)}")
    
    def _analyze_single_column(self, col_name) -> Dict[str, Any]:
        """
        Analyze a single column in the DataFrame.
        
        Args:
            col_name (str): Name of the column to analyze
            
        Returns:
            Dict[str, Any]: Details about the column
        """
        details = {
            "type": str(self.df[col_name].dtype)
        }
        
        # Get unique values count
        unique_count = self.df[col_name].nunique()
        print(f"Analyzing column '{col_name}' with {unique_count} unique values")
        details["unique_values"] = int(unique_count)
        details["unique_pct"] = float(unique_count / len(self.df) * 100)
        
        # Check if it's a categorical column (less than 20 or 20% unique values)
        is_categorical = unique_count <= min(20,len(self.df) * 0.2)
        print(f"Column '{col_name}' seems categorical: {is_categorical}")
        details["seems_categorical"] = is_categorical
        
        # If categorical, get top values
        if is_categorical and unique_count <= 30:  # Only if reasonable number of categories
            try:
                value_counts = self.df[col_name].value_counts().head(10).to_dict()
                # Convert keys to strings to ensure JSON compatibility
                details["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
            except:
                pass
        
        # If numeric, get statistics
        if pd.api.types.is_numeric_dtype(self.df[col_name]):
            try:
                non_na_vals = self.df[col_name].dropna()
                if len(non_na_vals) > 0:
                    details.update({
                        "min": float(non_na_vals.min()),
                        "max": float(non_na_vals.max()),
                        "mean": float(non_na_vals.mean()),
                        "median": float(non_na_vals.median()),
                        "std": float(non_na_vals.std())
                    })
                    
                    # Check for outliers (values more than 3 std devs from mean)
                    std = non_na_vals.std()
                    mean = non_na_vals.mean()
                    if not pd.isna(std) and not pd.isna(mean) and std > 0:
                        outliers = non_na_vals[(non_na_vals - mean).abs() > 3 * std]
                        details["outlier_count"] = int(len(outliers))
                        details["outlier_pct"] = float(len(outliers) / len(non_na_vals) * 100)
            except Exception as e:
                if self.verbose:
                    print(f"Error calculating statistics for {col_name}: {str(e)}")
        
        # Check if it might be a date
        if self.df[col_name].dtype == 'object':
            try:
                # Try to convert a sample to datetime
                sample = self.df[col_name].dropna().head(10)
                if len(sample) > 0:
                    # Use errors='coerce' to silence warnings and check if conversion worked
                    converted = pd.to_datetime(sample, errors='coerce')
                    # If most values converted successfully, it's likely a date
                    if converted.notna().sum() >= len(sample) * 0.8:  # 80% success rate
                        details["seems_like_date"] = True
                    else:
                        details["seems_like_date"] = False
            except:
                details["seems_like_date"] = False
        
        return details
    
    def _check_duplicates(self):
        """
        Check for duplicate rows in the DataFrame.
        """
        try:
            # Check for duplicate rows
            duplicates = self.df.duplicated()
            duplicate_count = duplicates.sum()
            self.basic_info["duplicate_rows"] = int(duplicate_count)
            self.basic_info["duplicate_pct"] = float(duplicate_count / len(self.df) * 100)
            
            # If there are duplicates, get a sample
            if duplicate_count > 0:
                duplicate_indices = duplicates[duplicates].index.tolist()
                sample_size = min(5, len(duplicate_indices))
                self.basic_info["duplicate_sample_indices"] = duplicate_indices[:sample_size]
        except Exception as e:
            if self.verbose:
                print(f"Error checking duplicates: {str(e)}")
    
    def _check_correlations(self):
        """
        Calculate correlations between numeric columns.
        """
        try:
            # Get numeric columns
            numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            
            # Only calculate correlations if there are at least 2 numeric columns
            if len(numeric_cols) >= 2:
                # Calculate correlation matrix
                corr_matrix = self.df[numeric_cols].corr().round(3)
                
                # Convert to dictionary format
                corr_dict = {}
                for col1 in numeric_cols:
                    corr_dict[col1] = {}
                    for col2 in numeric_cols:
                        if col1 != col2:  # Skip self-correlations
                            corr_dict[col1][col2] = float(corr_matrix.loc[col1, col2])
                
                # Find strong correlations (absolute value > 0.7)
                strong_correlations = []
                for col1 in numeric_cols:
                    for col2 in numeric_cols:
                        if col1 < col2:  # Avoid duplicates (correlation is symmetric)
                            corr = corr_matrix.loc[col1, col2]
                            if abs(corr) > 0.7:
                                strong_correlations.append({
                                    "column1": col1,
                                    "column2": col2,
                                    "correlation": float(corr)
                                })
                
                # Add correlations to basic info
                self.basic_info["correlations"] = corr_dict
                self.basic_info["strong_correlations"] = strong_correlations
        except Exception as e:
            if self.verbose:
                print(f"Error calculating correlations: {str(e)}")
                
    def get_column_summary(self, col_name: str) -> Dict[str, Any]:
        """
        Get a summary of a specific column.
        
        Args:
            col_name (str): Name of the column
            
        Returns:
            Dict[str, Any]: Summary of the column
        """
        if col_name not in self.df.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")
            
        if "column_details" not in self.basic_info or col_name not in self.basic_info["column_details"]:
            self._analyze_single_column(col_name)
            
        summary = {
            "name": col_name,
            "type": str(self.df[col_name].dtype),
            "missing_values": int(self.df[col_name].isna().sum()),
            "missing_pct": float(self.df[col_name].isna().sum() / len(self.df) * 100)
        }
        
        # Add details from column_details if available
        if "column_details" in self.basic_info and col_name in self.basic_info["column_details"]:
            summary.update(self.basic_info["column_details"][col_name])
            
        return summary
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary of the DataFrame analysis.
        
        Returns:
            Dict[str, Any]: Summary of DataFrame analysis
        """
        if not self.basic_info:
            self.analyze()
        
        summary = {
            "row_count": self.basic_info.get("row_count", 0),
            "column_count": self.basic_info.get("column_count", 0),
            "memory_usage": self.basic_info.get("memory_usage", "unknown"),
        }
        
        # Missing values summary
        if "total_missing_values" in self.basic_info:
            summary["missing_values"] = {
                "total": self.basic_info["total_missing_values"],
                "percentage": round(self.basic_info["total_missing_pct"], 2)
            }
            
            if "high_missing_columns" in self.basic_info:
                summary["missing_values"]["high_missing_columns"] = self.basic_info["high_missing_columns"]
        
        # Duplicate rows
        if "duplicate_rows" in self.basic_info:
            summary["duplicate_rows"] = {
                "count": self.basic_info["duplicate_rows"],
                "percentage": round(self.basic_info["duplicate_pct"], 2)
            }
        
        # Date columns
        if "date_columns" in self.basic_info:
            date_ranges = {}
            for col in self.basic_info["date_columns"]:
                min_key = f"{col}_min_date"
                max_key = f"{col}_max_date"
                if min_key in self.basic_info and max_key in self.basic_info:
                    date_ranges[col] = {
                        "min": self.basic_info[min_key],
                        "max": self.basic_info[max_key]
                    }
            
            if date_ranges:
                summary["date_columns"] = date_ranges
        
        # Column types summary
        if "column_types" in self.basic_info:
            type_counts = {}
            for col_type in set(self.basic_info["column_types"].values()):
                type_counts[col_type] = sum(1 for t in self.basic_info["column_types"].values() if t == col_type)
            
            summary["column_types"] = type_counts
        
        # Column categorizations
        if "column_details" in self.basic_info:
            categorical_cols = []
            numeric_cols = []
            date_cols = []
            text_cols = []
            
            for col, details in self.basic_info["column_details"].items():
                if details.get("seems_categorical", False):
                    categorical_cols.append(col)
                
                if "type" in details:
                    if "float" in details["type"] or "int" in details["type"]:
                        numeric_cols.append(col)
                    elif "datetime" in details["type"] or details.get("seems_like_date", False):
                        date_cols.append(col)
                    elif "object" in details["type"] and not details.get("seems_categorical", False):
                        text_cols.append(col)
            
            summary["column_categories"] = {
                "categorical": categorical_cols,
                "numeric": numeric_cols,
                "dates": date_cols,
                "text": text_cols
            }
        
        # Strong correlations
        if "strong_correlations" in self.basic_info and self.basic_info["strong_correlations"]:
            summary["strong_correlations"] = self.basic_info["strong_correlations"]
        
        return summary
    
    def print_summary(self):
        """
        Print a formatted summary of the DataFrame analysis.
        """
        summary = self.get_summary()
        
        print("\n" + "=" * 50)
        print(f"DATAFRAME SUMMARY")
        print("=" * 50)
        
        # Basic info
        print(f"\nRows: {summary['row_count']:,}")
        print(f"Columns: {summary['column_count']}")
        print(f"Memory Usage: {summary['memory_usage']}")
        
        # Missing values
        if "missing_values" in summary:
            print(f"\nMissing Values: {summary['missing_values']['total']:,} ({summary['missing_values']['percentage']}%)")
            
            if "high_missing_columns" in summary["missing_values"]:
                print(f"Columns with >20% missing values: {', '.join(summary['missing_values']['high_missing_columns'])}")
        
        # Duplicate rows
        if "duplicate_rows" in summary:
            print(f"\nDuplicate Rows: {summary['duplicate_rows']['count']:,} ({summary['duplicate_rows']['percentage']}%)")
        
        # Column types
        if "column_types" in summary:
            print("\nColumn Types:")
            for col_type, count in summary["column_types"].items():
                print(f"  - {col_type}: {count}")
        
        # Column categories
        if "column_categories" in summary:
            cats = summary["column_categories"]
            
            print("\nColumn Categories:")
            print(f"  - Categorical: {len(cats['categorical'])} columns")
            print(f"  - Numeric: {len(cats['numeric'])} columns")
            print(f"  - Dates: {len(cats['dates'])} columns")
            print(f"  - Text: {len(cats['text'])} columns")
        
        # Date ranges
        if "date_columns" in summary:
            print("\nDate Ranges:")
            for col, range_info in summary["date_columns"].items():
                print(f"  - {col}: {range_info['min']} to {range_info['max']}")
        
        # Strong correlations
        if "strong_correlations" in summary:
            print("\nStrong Correlations:")
            for corr in summary["strong_correlations"]:
                print(f"  - {corr['column1']} ↔ {corr['column2']}: {corr['correlation']:.3f}")
        
        print("\n" + "=" * 50)

    def get_analysis_for_next_stage(self) -> Dict[str, Any]:
        """
        Prepare analysis results in a structured format optimal for the next processing stage.
        This method returns a carefully organized dictionary with all analysis results,
        formatted specifically for easy consumption by downstream components.
        
        Returns:
            Dict[str, Any]: Structured analysis results
        """
        if not self.basic_info:
            self.analyze()
        
        # Structure the analysis results in a format optimized for next stages
        analysis_package = {}
        
        # 1. Basic DataFrame information
        analysis_package["dataframe_info"] = {
            "row_count": self.basic_info.get("row_count", 0),
            "column_count": self.basic_info.get("column_count", 0),
            "memory_usage": self.basic_info.get("memory_usage", "unknown"),
            "duplicate_rows": self.basic_info.get("duplicate_rows", 0),
            "duplicate_percentage": self.basic_info.get("duplicate_pct", 0)
        }
        
        # 2. Missing values information
        missing_values = self.basic_info.get("missing_values", {})
        missing_percentages = self.basic_info.get("missing_pct", {})
        
        analysis_package["missing_data"] = {
            "total_missing": self.basic_info.get("total_missing_values", 0),
            "total_missing_percentage": self.basic_info.get("total_missing_pct", 0),
            "columns_with_missing": {
                col: {
                    "count": missing_values.get(col, 0),
                    "percentage": missing_percentages.get(col, 0)
                }
                for col in missing_values.keys() if missing_values.get(col, 0) > 0
            },
            "high_missing_columns": self.basic_info.get("high_missing_columns", [])
        }
        
        # 3. Column categorization
        column_details = self.basic_info.get("column_details", {})
        
        categorical_cols = []
        numeric_cols = []
        datetime_cols = []
        text_cols = []
        id_cols = []
        
        for col, details in column_details.items():
            # Categorize columns by type for easier consumption by next stage
            if details.get("seems_like_date", False) or "datetime" in details.get("type", ""):
                datetime_cols.append(col)
            elif details.get("semantic_type", "") == "identifier" or col.lower().endswith('id'):
                id_cols.append(col)
            elif details.get("seems_categorical", False):
                categorical_cols.append(col)
            elif "float" in details.get("type", "") or "int" in details.get("type", ""):
                numeric_cols.append(col)
            elif "object" in details.get("type", "") and not details.get("seems_categorical", False):
                text_cols.append(col)
        
        analysis_package["column_categories"] = {
            "categorical": categorical_cols,
            "numeric": numeric_cols,
            "datetime": datetime_cols,
            "text": text_cols,
            "id": id_cols
        }
        
        # 4. Detailed column information
        column_info = {}
        for col, details in column_details.items():
            # Clean up and standardize column details
            col_info = {
                "name": col,
                "type": details.get("type", "unknown"),
                "semantic_type": details.get("semantic_type", "unknown"),
                "missing_count": missing_values.get(col, 0),
                "missing_percentage": missing_percentages.get(col, 0),
                "unique_count": details.get("unique_values", 0),
                "unique_percentage": details.get("unique_pct", 0),
                "is_categorical": details.get("seems_categorical", False),
                "is_datetime": details.get("seems_like_date", False) or "datetime" in details.get("type", ""),
                "is_numeric": "float" in details.get("type", "") or "int" in details.get("type", ""),
                "is_id": details.get("semantic_type", "") == "identifier" or col.lower().endswith('id'),
            }
            
            # Add statistics for numeric columns
            if col_info["is_numeric"]:
                col_info.update({
                    "min": details.get("min", None),
                    "max": details.get("max", None),
                    "mean": details.get("mean", None),
                    "median": details.get("median", None),
                    "std": details.get("std", None),
                    "has_outliers": (details.get("outlier_count", 0) > 0),
                    "outlier_count": details.get("outlier_count", 0),
                    "outlier_percentage": details.get("outlier_pct", 0)
                })
            
            # Add top values for categorical columns
            if col_info["is_categorical"] and "top_values" in details:
                col_info["top_values"] = details["top_values"]
            
            # Add date range for datetime columns
            if col_info["is_datetime"]:
                min_key = f"{col}_min_date"
                max_key = f"{col}_max_date"
                if min_key in self.basic_info and max_key in self.basic_info:
                    col_info["date_range"] = {
                        "min": self.basic_info[min_key],
                        "max": self.basic_info[max_key]
                    }
            
            column_info[col] = col_info
        
        analysis_package["column_details"] = column_info
        
        # 5. Correlation information
        if "strong_correlations" in self.basic_info:
            analysis_package["correlations"] = {
                "strong_correlations": self.basic_info["strong_correlations"],
                # Provide a simplified version of correlations for easy lookup
                "correlation_matrix": self.basic_info.get("correlations", {})
            }
        
        return analysis_package

class ColumnDescriptionParser:
    """
    Parses and enhances column descriptions from various input formats.
    This class handles user-provided column descriptions and enriches them
    with insights from data analysis.
    """
    
    def __init__(self, col_desc_str=None, analysis_results=None, verbose=True):
        """
        Initialize the column description parser.
        
        Args:
            col_desc_str (str, optional): Column descriptions as a string
            analysis_results (Dict[str, Any], optional): Results from DataFrameAnalyzer
            verbose (bool): Whether to print progress information
        """
        self.col_desc_str = col_desc_str or ""
        self.analysis_results = analysis_results or {}
        self.verbose = verbose
        self.col_descriptions = {}
        self.enhanced_descriptions = {}
        self.total_rows = 0  # Initialize total_rows with default value
        
        # Parse descriptions if provided
        if col_desc_str:
            self.parse()
            
        # Enhance descriptions if analysis results provided
        if analysis_results and self.col_descriptions:
            self.enhance_descriptions()
    
    def parse(self, col_desc_str=None) -> Dict[str, str]:
        """
        Parse column descriptions from a string format.
        Handles various input formats including multi-line strings,
        with column name and description separated by a colon.
        
        Args:
            col_desc_str (str, optional): Column descriptions as a string
            
        Returns:
            Dict[str, str]: Dictionary of column descriptions
        """
        if col_desc_str is not None:
            self.col_desc_str = col_desc_str
        
        # Reset descriptions
        self.col_descriptions = {}
        
        try:
            if not self.col_desc_str.strip():
                return self.col_descriptions
            
            # Split the input string into lines and process each line
            lines = self.col_desc_str.strip().split('\n')
            current_col = None
            current_desc = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line starts a new column description
                if ':' in line and not current_desc:
                    parts = line.split(':', 1)
                    current_col = parts[0].strip()
                    desc = parts[1].strip()
                    
                    if current_col and desc:
                        # This is a single-line description
                        self.col_descriptions[current_col] = desc
                        current_col = None
                    elif current_col:
                        # This is the start of a multi-line description
                        current_desc = [desc] if desc else []
                else:
                    # This is a continuation of a multi-line description
                    if current_col is not None:
                        current_desc.append(line)
            
            # Add the last multi-line description if there is one
            if current_col is not None and current_desc:
                self.col_descriptions[current_col] = ' '.join(current_desc)
                
            if self.verbose:
                print(f"Parsed {len(self.col_descriptions)} column descriptions")
                
        except Exception as e:
            if self.verbose:
                print(f"Error parsing column descriptions: {str(e)}")
        
        return self.col_descriptions
    
    def enhance_descriptions(self, analysis_results=None) -> Dict[str, Dict[str, Any]]:
        """
        Enhance column descriptions with insights from data analysis.
        
        Args:
            analysis_results (Dict[str, Any], optional): Results from DataFrameAnalyzer
            
        Returns:
            Dict[str, Dict[str, Any]]: Enhanced column descriptions
        """
        if analysis_results is not None:
            self.analysis_results = analysis_results
        
        # Reset enhanced descriptions
        self.enhanced_descriptions = {}
        
        if not self.analysis_results or not self.col_descriptions:
            return self.enhanced_descriptions
        
        try:
            # Get column details from analysis results
            column_details = self.analysis_results.get("column_details", {})
            
            # Get total row count from analysis results
            self.total_rows = self.analysis_results.get("total_rows", 0)
            
            # If total_rows is not provided in analysis_results, try to determine it from data
            if self.total_rows == 0:
                # Check if we can find a row count from any column's info
                for col_name, col_info in column_details.items():
                    if "row_count" in col_info:
                        self.total_rows = col_info["row_count"]
                        break
            
            # If still not found, estimate from the first column with top_values
            if self.total_rows == 0:
                for col_name, col_info in column_details.items():
                    if col_info.get("is_categorical", False) and "top_values" in col_info:
                        top_values = col_info["top_values"]
                        total_count = 0
                        for value, count in top_values.items():
                            try:
                                count_numeric = int(count) if isinstance(count, str) else count
                                total_count += count_numeric
                            except (ValueError, TypeError):
                                pass
                        if total_count > 0:
                            self.total_rows = total_count
                            break
            
            if self.verbose:
                if self.total_rows > 0:
                    print(f"Dataset has {self.total_rows} total rows")
                else:
                    print("Warning: Could not determine total rows, some percentage calculations may be incorrect")
                    # Set to 1 to avoid division by zero
                    self.total_rows = 1
            
            # Process each column with a description
            for col_name, user_desc in self.col_descriptions.items():
                # Start with the user description
                enhanced = {
                    "name": col_name,
                    "user_description": user_desc,
                    "has_user_description": True
                }
                
                # Add analysis insights if available
                if col_name in column_details:
                    col_info = column_details[col_name]
                    
                    # Add basic column info
                    enhanced.update({
                        "type": col_info.get("type", "unknown"),
                        "is_categorical": col_info.get("is_categorical", False),
                        "is_numeric": col_info.get("is_numeric", False),
                        "is_datetime": col_info.get("is_datetime", False),
                        "is_id": col_info.get("is_id", False),
                        "missing_count": col_info.get("missing_count", 0),
                        "missing_percentage": col_info.get("missing_percentage", 0)
                    })
                    
                    # Add type-specific information
                    if col_info.get("is_numeric", False):
                        enhanced["numeric_info"] = {
                            "min": col_info.get("min"),
                            "max": col_info.get("max"),
                            "mean": col_info.get("mean"),
                            "median": col_info.get("median"),
                            "std": col_info.get("std"),
                            "has_outliers": col_info.get("has_outliers", False)
                        }
                    
                    # Enhanced categorical information
                    if col_info.get("is_categorical", False):
                        unique_count = col_info.get("unique_count", 0)
                        top_values = col_info.get("top_values", {})
                        
                        # Ensure all values in top_values are numeric
                        top_values_numeric = {}
                        for k, v in top_values.items():
                            try:
                                top_values_numeric[k] = int(v) if isinstance(v, str) else v
                            except (ValueError, TypeError):
                                top_values_numeric[k] = 0
                                if self.verbose:
                                    print(f"Warning: Could not convert value count '{v}' to int for '{k}' in column {col_name}")
                        
                        # Create sorted list of top values with numeric counts
                        sorted_top_values = []
                        if top_values_numeric:
                            sorted_top_values = sorted(
                                [(k, v) for k, v in top_values_numeric.items()], 
                                key=lambda x: x[1], 
                                reverse=True
                            )
                        
                        # Make sure we have numeric values for the counts
                        most_common_value = sorted_top_values[0][0] if sorted_top_values else None
                        most_common_count = int(sorted_top_values[0][1]) if sorted_top_values else 0
                        
                        # Calculate total and percentages
                        total_top_values = sum(top_values_numeric.values())
                        top_value_percentage = float(total_top_values) / self.total_rows * 100 if self.total_rows > 0 else 0
                        most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 and most_common_count > 0 else 0
                        
                        enhanced["categorical_info"] = {
                            "unique_count": unique_count,
                            "top_values": top_values_numeric,
                            "sorted_top_values": sorted_top_values,
                            "top_value_counts": len(top_values),
                            "top_value_percentage": top_value_percentage,
                            "most_common": most_common_value,
                            "most_common_count": most_common_count,
                            "most_common_percentage": most_common_percentage
                        }
                    
                    if col_info.get("is_datetime", False) and "date_range" in col_info:
                        enhanced["datetime_info"] = {
                            "date_range": col_info.get("date_range", {})
                        }
                
                # Generate a combined description
                combined_desc = self._generate_combined_description(col_name, enhanced)
                enhanced["combined_description"] = combined_desc
                
                # Add to enhanced descriptions
                self.enhanced_descriptions[col_name] = enhanced
            
            # For columns without user descriptions but with analysis data,
            # create placeholder enhanced descriptions
            for col_name, col_info in column_details.items():
                if col_name not in self.enhanced_descriptions:
                    enhanced = {
                        "name": col_name,
                        "user_description": "",
                        "has_user_description": False,
                        "type": col_info.get("type", "unknown"),
                        "is_categorical": col_info.get("is_categorical", False),
                        "is_numeric": col_info.get("is_numeric", False),
                        "is_datetime": col_info.get("is_datetime", False),
                        "is_id": col_info.get("is_id", False),
                        "missing_count": col_info.get("missing_count", 0),
                        "missing_percentage": col_info.get("missing_percentage", 0)
                    }
                    
                    # Add type-specific information
                    if col_info.get("is_numeric", False):
                        enhanced["numeric_info"] = {
                            "min": col_info.get("min"),
                            "max": col_info.get("max"),
                            "mean": col_info.get("mean"),
                            "median": col_info.get("median"),
                            "std": col_info.get("std"),
                            "has_outliers": col_info.get("has_outliers", False)
                        }
                    
                    # Enhanced categorical information
                    if col_info.get("is_categorical", False):
                        unique_count = col_info.get("unique_count", 0)
                        top_values = col_info.get("top_values", {})
                        
                        # Ensure all values in top_values are numeric
                        top_values_numeric = {}
                        for k, v in top_values.items():
                            try:
                                top_values_numeric[k] = int(v) if isinstance(v, str) else v
                            except (ValueError, TypeError):
                                top_values_numeric[k] = 0
                                if self.verbose:
                                    print(f"Warning: Could not convert value count '{v}' to int for '{k}' in column {col_name}")
                        
                        # Create sorted list of top values with numeric counts
                        sorted_top_values = []
                        if top_values_numeric:
                            sorted_top_values = sorted(
                                [(k, v) for k, v in top_values_numeric.items()], 
                                key=lambda x: x[1], 
                                reverse=True
                            )
                        
                        # Make sure we have numeric values for the counts
                        most_common_value = sorted_top_values[0][0] if sorted_top_values else None
                        most_common_count = int(sorted_top_values[0][1]) if sorted_top_values else 0
                        
                        # Calculate total and percentages
                        total_top_values = sum(top_values_numeric.values())
                        top_value_percentage = float(total_top_values) / self.total_rows * 100 if self.total_rows > 0 else 0
                        most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 and most_common_count > 0 else 0
                        
                        enhanced["categorical_info"] = {
                            "unique_count": unique_count,
                            "top_values": top_values_numeric,
                            "sorted_top_values": sorted_top_values,
                            "top_value_counts": len(top_values),
                            "top_value_percentage": top_value_percentage,
                            "most_common": most_common_value,
                            "most_common_count": most_common_count,
                            "most_common_percentage": most_common_percentage
                        }
                    
                    if col_info.get("is_datetime", False) and "date_range" in col_info:
                        enhanced["datetime_info"] = {
                            "date_range": col_info.get("date_range", {})
                        }
                    
                    # Generate a combined description
                    combined_desc = self._generate_combined_description(col_name, enhanced)
                    enhanced["combined_description"] = combined_desc
                    
                    # Add to enhanced descriptions
                    self.enhanced_descriptions[col_name] = enhanced
            
            if self.verbose:
                print(f"Enhanced {len(self.enhanced_descriptions)} column descriptions")
                
        except Exception as e:
            if self.verbose:
                print(f"Error enhancing descriptions: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return self.enhanced_descriptions
    
    def _generate_combined_description(self, col_name: str, enhanced_info: Dict[str, Any]) -> str:
        """
        Generate a combined description using user input and analysis insights.
        
        Args:
            col_name (str): Name of the column
            enhanced_info (Dict[str, Any]): Enhanced column information
            
        Returns:
            str: Combined description
        """
        parts = []
        
        # Start with user description if available
        if enhanced_info.get("has_user_description", False):
            parts.append(enhanced_info["user_description"])
        
        # Add type information
        col_type = "unknown"
        if enhanced_info.get("is_id", False):
            col_type = "identifier"
        elif enhanced_info.get("is_datetime", False):
            col_type = "date/time"
        elif enhanced_info.get("is_categorical", False):
            col_type = "categorical"
        elif enhanced_info.get("is_numeric", False):
            col_type = "numeric"
        elif "type" in enhanced_info:
            col_type = enhanced_info["type"]
        
        parts.append(f"This is a {col_type} column.")
        
        # Add missing value information if significant
        missing_pct = enhanced_info.get("missing_percentage", 0)
        if missing_pct > 5:  # Only mention if above 5%
            parts.append(f"It has {missing_pct:.1f}% missing values.")
        
        # Add type-specific insights
        if enhanced_info.get("is_numeric", False) and "numeric_info" in enhanced_info:
            num_info = enhanced_info["numeric_info"]
            # Format numeric values nicely
            min_val = self._format_number(num_info.get("min"))
            max_val = self._format_number(num_info.get("max"))
            mean_val = self._format_number(num_info.get("mean"))
            
            parts.append(f"Values range from {min_val} to {max_val}, with an average of {mean_val}.")
            
            if num_info.get("has_outliers", False):
                parts.append("It contains some outlier values.")
        
        # Enhanced categorical information with more details
        if enhanced_info.get("is_categorical", False) and "categorical_info" in enhanced_info:
            cat_info = enhanced_info["categorical_info"]
            unique_count = cat_info.get("unique_count", 0)
            
            if unique_count > 0:
                parts.append(f"It contains {unique_count} unique categories.")
            
            # Include most common value and its frequency
            most_common = cat_info.get("most_common")
            most_common_pct = cat_info.get("most_common_percentage", 0)
            if most_common:
                parts.append(f"The most common value is '{most_common}', appearing in approximately {most_common_pct:.1f}% of the data.")
            
            # Include all top categories with counts and percentages
            sorted_top_values = cat_info.get("sorted_top_values", [])
            if len(sorted_top_values) > 1:  # If there are multiple top values
                top_values_str = []
                for value, count in sorted_top_values[:5]:  # Limit to top 5 for readability
                    # Ensure count is numeric
                    try:
                        count_numeric = int(count) if isinstance(count, str) else count
                        percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
                        top_values_str.append(f"'{value}' ({count_numeric} occurrences, {percentage:.1f}%)")
                    except (ValueError, TypeError):
                        top_values_str.append(f"'{value}' ({count} occurrences)")
                
                if len(sorted_top_values) > 5:
                    top_values_list = ", ".join(top_values_str)
                    parts.append(f"Top categories include: {top_values_list}, and {len(sorted_top_values) - 5} more.")
                else:
                    top_values_list = ", ".join(top_values_str)
                    parts.append(f"Top categories include: {top_values_list}.")
        
        if enhanced_info.get("is_datetime", False) and "datetime_info" in enhanced_info:
            date_range = enhanced_info["datetime_info"].get("date_range", {})
            if "min" in date_range and "max" in date_range:
                parts.append(f"Dates span from {date_range['min']} to {date_range['max']}.")
        
        return " ".join(parts)
    
    def _format_number(self, value) -> str:
        """
        Format a number for display in descriptions.
        
        Args:
            value: Numeric value to format
            
        Returns:
            str: Formatted number
        """
        if value is None:
            return "unknown"
        
        try:
            # Integer formatting
            if value == int(value):
                return f"{int(value):,}"
            
            # Float formatting - different precision based on magnitude
            abs_val = abs(value)
            if abs_val >= 1000:
                return f"{value:,.0f}"
            elif abs_val >= 100:
                return f"{value:,.1f}"
            elif abs_val >= 1:
                return f"{value:,.2f}"
            elif abs_val == 0:
                return "0"
            else:
                # For very small numbers, use scientific notation
                return f"{value:.2e}"
        except:
            return str(value)
    
    def get_descriptions_dict(self) -> Dict[str, str]:
        """
        Get the parsed column descriptions as a simple dictionary.
        
        Returns:
            Dict[str, str]: Dictionary mapping column names to descriptions
        """
        return self.col_descriptions
    
    def get_enhanced_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the enhanced column descriptions.
        
        Returns:
            Dict[str, Dict[str, Any]]: Enhanced column descriptions
        """
        return self.enhanced_descriptions
    
    def get_combined_descriptions(self) -> Dict[str, str]:
        """
        Get the combined descriptions (user + analysis insights).
        
        Returns:
            Dict[str, str]: Dictionary mapping column names to combined descriptions
        """
        combined = {}
        for col_name, info in self.enhanced_descriptions.items():
            combined[col_name] = info.get("combined_description", "")
        return combined
    
    def print_descriptions(self):
        """
        Print the parsed column descriptions.
        """
        if not self.col_descriptions:
            print("No column descriptions available.")
            return
            
        print("\n" + "=" * 50)
        print("COLUMN DESCRIPTIONS")
        print("=" * 50)
        
        for col, desc in sorted(self.col_descriptions.items()):
            print(f"\n{col}:")
            print(f"  {desc}")
    
    def print_enhanced_descriptions(self):
        """
        Print the enhanced column descriptions.
        """
        if not self.enhanced_descriptions:
            print("No enhanced descriptions available.")
            return
            
        print("\n" + "=" * 50)
        print("ENHANCED COLUMN DESCRIPTIONS")
        print("=" * 50)
        
        for col, info in sorted(self.enhanced_descriptions.items()):
            print(f"\n{col}:")
            if info.get("has_user_description", False):
                print(f"  User Description: {info['user_description']}")
            print(f"  Combined Description: {info['combined_description']}")
            
            # Print type information
            col_type = []
            if info.get("is_id", False): col_type.append("ID")
            if info.get("is_categorical", False): col_type.append("Categorical")
            if info.get("is_numeric", False): col_type.append("Numeric")
            if info.get("is_datetime", False): col_type.append("Date/Time")
            
            print(f"  Type: {', '.join(col_type) if col_type else info.get('type', 'Unknown')}")
            
            # Print statistics for numeric columns
            if info.get("is_numeric", False) and "numeric_info" in info:
                num_info = info["numeric_info"]
                print(f"  Range: {self._format_number(num_info.get('min'))} to {self._format_number(num_info.get('max'))}")
                print(f"  Mean: {self._format_number(num_info.get('mean'))}, Median: {self._format_number(num_info.get('median'))}")
            
            # Enhanced categorical information display
            if info.get("is_categorical", False) and "categorical_info" in info:
                cat_info = info["categorical_info"]
                print(f"  Unique Categories: {cat_info.get('unique_count', 0)}")
                
                # Show most common value
                if cat_info.get("most_common") is not None:
                    try:
                        most_common_count = cat_info.get("most_common_count", 0)
                        # Ensure most_common_count is numeric
                        if isinstance(most_common_count, str):
                            most_common_count = int(most_common_count)
                        print(f"  Most Common: '{cat_info.get('most_common')}' ({most_common_count} occurrences, {cat_info.get('most_common_percentage', 0):.1f}%)")
                    except (ValueError, TypeError):
                        print(f"  Most Common: '{cat_info.get('most_common')}' (error calculating stats)")
                
                # Show all top values with counts and percentages
                print("  Top Categories:")
                sorted_top_values = cat_info.get("sorted_top_values", [])
                for i, (value, count) in enumerate(sorted_top_values[:10]):  # Show up to 10 top values
                    try:
                        # Ensure count is numeric
                        count_numeric = int(count) if isinstance(count, str) else count
                        percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
                        print(f"    {i+1}. '{value}': {count_numeric} occurrences ({percentage:.1f}%)")
                    except (ValueError, TypeError):
                        print(f"    {i+1}. '{value}': {count} occurrences (error calculating percentage)")
                
                if len(sorted_top_values) > 10:
                    print(f"    ... and {len(sorted_top_values) - 10} more categories")
            
            # Print date range for datetime columns
            if info.get("is_datetime", False) and "datetime_info" in info:
                date_range = info["datetime_info"].get("date_range", {})
                if date_range:
                    print(f"  Date Range: {date_range.get('min', 'Unknown')} to {date_range.get('max', 'Unknown')}")
    
    def get_categorical_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a summary of all categorical columns with detailed category information.
        
        Returns:
            Dict[str, Dict[str, Any]]: Summary of categorical columns
        """
        categorical_summary = {}
        
        # Check if total_rows is set, if not try to get it from analysis_results
        if self.total_rows == 0 and self.analysis_results:
            self.total_rows = self.analysis_results.get("total_rows", 0)
            if self.verbose and self.total_rows > 0:
                print(f"Setting total_rows to {self.total_rows} in get_categorical_summary")
        
        # If still zero, try to calculate from the data
        if self.total_rows == 0:
            # Try to estimate total rows from the first categorical column's data
            for col_info in self.enhanced_descriptions.values():
                if col_info.get("is_categorical", False) and "categorical_info" in col_info:
                    cat_info = col_info["categorical_info"]
                    sorted_top_values = cat_info.get("sorted_top_values", [])
                    if sorted_top_values:
                        total_count = 0
                        for _, count in sorted_top_values:
                            try:
                                count_numeric = int(count) if isinstance(count, str) else count
                                total_count += count_numeric
                            except (ValueError, TypeError):
                                pass
                        if total_count > 0:
                            self.total_rows = total_count
                            if self.verbose:
                                print(f"Estimated total_rows as {self.total_rows} from category counts")
                            break
        
        # Default to 1 to avoid division by zero if we couldn't determine total_rows
        if self.total_rows == 0:
            self.total_rows = 1
            if self.verbose:
                print("Warning: Could not determine total_rows, using default value of 1")
                
        for col_name, info in self.enhanced_descriptions.items():
            if info.get("is_categorical", False) and "categorical_info" in info:
                cat_info = info["categorical_info"]
                
                # Make sure most_common_count is numeric
                most_common_count = 0
                try:
                    most_common_count_val = cat_info.get("most_common_count", 0)
                    most_common_count = int(most_common_count_val) if isinstance(most_common_count_val, str) else most_common_count_val
                except (ValueError, TypeError):
                    if self.verbose:
                        print(f"Warning: Could not convert most_common_count to int for column {col_name}")
                
                # Calculate most_common_percentage
                most_common_percentage = 0
                try:
                    most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 else 0
                except (ValueError, TypeError):
                    if self.verbose:
                        print(f"Warning: Error calculating most_common_percentage for column {col_name}")
                
                summary = {
                    "column_name": col_name,
                    "description": info.get("user_description", ""),
                    "combined_description": info.get("combined_description", ""),
                    "unique_categories": cat_info.get("unique_count", 0),
                    "most_common": cat_info.get("most_common"),
                    "most_common_count": most_common_count,
                    "most_common_percentage": most_common_percentage,
                    "categories": []
                }
                
                # Add all categories with their counts and percentages
                for value, count in cat_info.get("sorted_top_values", []):
                    try:
                        # Ensure count is numeric
                        count_numeric = int(count) if isinstance(count, str) else count
                        percentage = float(count_numeric) / self.total_rows * 100 if self.total_rows > 0 else 0
                        summary["categories"].append({
                            "value": value,
                            "count": count_numeric,
                            "percentage": percentage
                        })
                    except (ValueError, TypeError):
                        if self.verbose:
                            print(f"Warning: Error processing category '{value}' for column {col_name}")
                
                categorical_summary[col_name] = summary
        
        return categorical_summary

    def get_output_for_next_stage(self) -> Dict[str, Any]:
        """
        Prepare a comprehensive output package for the next processing stage.
        This method returns a carefully organized dictionary with all column descriptions,
        interpretations, and metadata, formatted for easy consumption by downstream components.
        
        Returns:
            Dict[str, Any]: Structured output for next stage
        """
        # Make sure we have enhanced descriptions
        if not self.enhanced_descriptions:
            # Try to generate enhanced descriptions if we have analysis results
            if self.analysis_results:
                self.enhance_descriptions()
            else:
                if self.verbose:
                    print("No analysis results available. Cannot generate enhanced descriptions.")
                return {}
            
            # Double-check that we now have enhanced descriptions
            if not self.enhanced_descriptions:
                if self.verbose:
                    print("Unable to generate enhanced descriptions.")
                return {}
        
        output_package = {
            "metadata": {
                "total_columns": len(self.enhanced_descriptions),
                "columns_with_user_descriptions": sum(1 for info in self.enhanced_descriptions.values() 
                                                    if info.get("has_user_description", False)),
                "categorical_columns": sum(1 for info in self.enhanced_descriptions.values() 
                                            if info.get("is_categorical", False)),
                "numeric_columns": sum(1 for info in self.enhanced_descriptions.values() 
                                        if info.get("is_numeric", False)),
                "datetime_columns": sum(1 for info in self.enhanced_descriptions.values() 
                                        if info.get("is_datetime", False)),
                "id_columns": sum(1 for info in self.enhanced_descriptions.values() 
                                if info.get("is_id", False))
            },
            "columns": {},
            "text_descriptions": {},
            "column_groups": {
                "categorical": [],
                "numeric": [],
                "datetime": [],
                "id": [],
                "text": [],
                "with_user_description": [],
                "without_user_description": []
            }
        }
        
        # Process each column
        for col_name, info in self.enhanced_descriptions.items():
            # Add to column groups
            if info.get("is_categorical", False):
                output_package["column_groups"]["categorical"].append(col_name)
            if info.get("is_numeric", False):
                output_package["column_groups"]["numeric"].append(col_name)
            if info.get("is_datetime", False):
                output_package["column_groups"]["datetime"].append(col_name)
            if info.get("is_id", False):
                output_package["column_groups"]["id"].append(col_name)
            if not any([info.get("is_categorical", False), 
                        info.get("is_numeric", False),
                        info.get("is_datetime", False),
                        info.get("is_id", False)]):
                output_package["column_groups"]["text"].append(col_name)
            
            if info.get("has_user_description", False):
                output_package["column_groups"]["with_user_description"].append(col_name)
            else:
                output_package["column_groups"]["without_user_description"].append(col_name)
            
            # Add text descriptions
            output_package["text_descriptions"][col_name] = {
                "user_description": info.get("user_description", ""),
                "combined_description": info.get("combined_description", "")
            }
            
            # Add structured column information
            column_data = {
                "name": col_name,
                "has_user_description": info.get("has_user_description", False),
                "type": info.get("type", "unknown"),
                "is_categorical": info.get("is_categorical", False),
                "is_numeric": info.get("is_numeric", False),
                "is_datetime": info.get("is_datetime", False),
                "is_id": info.get("is_id", False),
                "missing_count": info.get("missing_count", 0),
                "missing_percentage": info.get("missing_percentage", 0)
            }
            
            # Add type-specific information
            if info.get("is_numeric", False) and "numeric_info" in info:
                column_data["numeric_stats"] = info["numeric_info"]
            
            if info.get("is_categorical", False) and "categorical_info" in info:
                # Ensure counts are numeric
                categories = []
                try:
                    for val, count in info["categorical_info"].get("sorted_top_values", []):
                        try:
                            count_numeric = int(count) if isinstance(count, str) else count
                            percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
                            categories.append({
                                "value": val,
                                "count": count_numeric,
                                "percentage": percentage
                            })
                        except (ValueError, TypeError):
                            if self.verbose:
                                print(f"Warning: Error processing category '{val}' in get_output_for_next_stage")
                    
                    most_common_count = info["categorical_info"].get("most_common_count", 0)
                    if isinstance(most_common_count, str):
                        most_common_count = int(most_common_count)
                    
                    column_data["categorical_stats"] = {
                        "unique_count": info["categorical_info"].get("unique_count", 0),
                        "most_common": info["categorical_info"].get("most_common"),
                        "most_common_count": most_common_count,
                        "most_common_percentage": info["categorical_info"].get("most_common_percentage", 0),
                        "categories": categories
                    }
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating categorical stats for {col_name}: {str(e)}")
                    # Provide a minimal version of categorical stats to avoid failures
                    column_data["categorical_stats"] = {
                        "unique_count": info["categorical_info"].get("unique_count", 0),
                        "most_common": info["categorical_info"].get("most_common"),
                        "most_common_count": 0,
                        "most_common_percentage": 0,
                        "categories": []
                    }
            
            if info.get("is_datetime", False) and "datetime_info" in info:
                column_data["datetime_stats"] = info["datetime_info"]
            
            output_package["columns"][col_name] = column_data
        
        return output_package

    def get_markdown_report(self) -> str:
        """
        Generate a comprehensive Markdown report of the column descriptions.
        
        Returns:
            str: Markdown-formatted report
        """
        # Make sure we have enhanced descriptions
        if not self.enhanced_descriptions:
            # Try to generate enhanced descriptions if we have analysis results
            if self.analysis_results:
                self.enhance_descriptions()
            else:
                return "No analysis results available. Cannot generate report."
            
            # Double-check that we now have enhanced descriptions
            if not self.enhanced_descriptions:
                return "Unable to generate enhanced descriptions for report."
        
        lines = []
        
        # Title
        lines.append("# Column Descriptions Report\n")
        
        # Summary statistics
        categorical_columns = [col for col, info in self.enhanced_descriptions.items() 
                                if info.get("is_categorical", False)]
        numeric_columns = [col for col, info in self.enhanced_descriptions.items() 
                            if info.get("is_numeric", False)]
        datetime_columns = [col for col, info in self.enhanced_descriptions.items() 
                            if info.get("is_datetime", False)]
        id_columns = [col for col, info in self.enhanced_descriptions.items() 
                        if info.get("is_id", False)]
        user_described = [col for col, info in self.enhanced_descriptions.items() 
                            if info.get("has_user_description", False)]
        
        lines.append("## Summary\n")
        lines.append(f"- **Total Columns:** {len(self.enhanced_descriptions)}")
        lines.append(f"- **Columns with User Descriptions:** {len(user_described)}")
        lines.append(f"- **Categorical Columns:** {len(categorical_columns)}")
        lines.append(f"- **Numeric Columns:** {len(numeric_columns)}")
        lines.append(f"- **Datetime Columns:** {len(datetime_columns)}")
        lines.append(f"- **ID Columns:** {len(id_columns)}")
        lines.append("")
        
        # Column List by Type
        if categorical_columns:
            lines.append("### Categorical Columns")
            for col in sorted(categorical_columns):
                lines.append(f"- {col}")
            lines.append("")
        
        if numeric_columns:
            lines.append("### Numeric Columns")
            for col in sorted(numeric_columns):
                lines.append(f"- {col}")
            lines.append("")
        
        if datetime_columns:
            lines.append("### Datetime Columns")
            for col in sorted(datetime_columns):
                lines.append(f"- {col}")
            lines.append("")
        
        if id_columns:
            lines.append("### ID Columns")
            for col in sorted(id_columns):
                lines.append(f"- {col}")
            lines.append("")
        
        # Detailed Column Descriptions
        lines.append("## Detailed Column Descriptions\n")
        
        for col_name in sorted(self.enhanced_descriptions.keys()):
            info = self.enhanced_descriptions[col_name]
            
            # Column header with type
            col_type = []
            if info.get("is_id", False): col_type.append("ID")
            if info.get("is_categorical", False): col_type.append("Categorical")
            if info.get("is_numeric", False): col_type.append("Numeric")
            if info.get("is_datetime", False): col_type.append("Date/Time")
            
            type_str = ", ".join(col_type) if col_type else info.get("type", "Unknown")
            lines.append(f"### {col_name} ({type_str})\n")
            
            # User description if available
            if info.get("has_user_description", False):
                lines.append(f"**User Description:** {info['user_description']}\n")
            
            # Combined description
            lines.append(f"**Description:** {info.get('combined_description', '')}\n")
            
            # Missing values if significant
            missing_pct = info.get("missing_percentage", 0)
            if missing_pct > 0:
                lines.append(f"**Missing Values:** {info.get('missing_count', 0)} ({missing_pct:.1f}%)\n")
            
            # Type-specific details
            if info.get("is_numeric", False) and "numeric_info" in info:
                num_info = info["numeric_info"]
                lines.append("**Numeric Statistics:**")
                lines.append(f"- **Range:** {self._format_number(num_info.get('min'))} to {self._format_number(num_info.get('max'))}")
                lines.append(f"- **Mean:** {self._format_number(num_info.get('mean'))}")
                lines.append(f"- **Median:** {self._format_number(num_info.get('median'))}")
                lines.append(f"- **Standard Deviation:** {self._format_number(num_info.get('std'))}")
                
                if num_info.get("has_outliers", False):
                    lines.append("- **Contains outliers**")
                lines.append("")
            
            if info.get("is_categorical", False) and "categorical_info" in info:
                cat_info = info["categorical_info"]
                lines.append("**Categorical Statistics:**")
                lines.append(f"- **Unique Categories:** {cat_info.get('unique_count', 0)}")
                
                most_common = cat_info.get("most_common")
                if most_common is not None:
                    try:
                        most_common_count = cat_info.get("most_common_count", 0)
                        if isinstance(most_common_count, str):
                            most_common_count = int(most_common_count)
                        lines.append(f"- **Most Common Value:** '{most_common}' ({most_common_count} occurrences, {cat_info.get('most_common_percentage', 0):.1f}%)")
                    except (ValueError, TypeError):
                        lines.append(f"- **Most Common Value:** '{most_common}'")
                
                sorted_top_values = cat_info.get("sorted_top_values", [])
                if sorted_top_values:
                    lines.append("\n**Top Categories:**")
                    
                    # Create markdown table for top categories
                    lines.append("| Category | Count | Percentage |")
                    lines.append("|---------|-------|------------|")
                    
                    for value, count in sorted_top_values[:10]:  # Limit to top 10
                        try:
                            # Ensure count is numeric
                            count_numeric = int(count) if isinstance(count, str) else count
                            percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
                            lines.append(f"| '{value}' | {count_numeric} | {percentage:.1f}% |")
                        except (ValueError, TypeError):
                            lines.append(f"| '{value}' | {count} | - |")
                    
                    if len(sorted_top_values) > 10:
                        lines.append(f"\n*...and {len(sorted_top_values) - 10} more categories*")
                
                lines.append("")
            
            if info.get("is_datetime", False) and "datetime_info" in info:
                date_range = info["datetime_info"].get("date_range", {})
                if date_range:
                    lines.append("**Date Range:**")
                    lines.append(f"- **From:** {date_range.get('min', 'Unknown')}")
                    lines.append(f"- **To:** {date_range.get('max', 'Unknown')}")
                    lines.append("")
            
            # Separator between columns
            lines.append("---\n")
        
        return "\n".join(lines)
    
    def get_llm_context_report(self, include_statistics=True, max_categories_per_column=5) -> str:
        """
        Generate a concise but comprehensive Markdown report of the column descriptions,
        optimized specifically for sending to an LLM as context.
        
        Args:
            include_statistics (bool): Whether to include detailed statistics in the report
            max_categories_per_column (int): Maximum number of top categories to include per categorical column
            
        Returns:
            str: Markdown-formatted report optimized for LLM context
        """
        # Make sure we have enhanced descriptions
        if not self.enhanced_descriptions:
            # Try to generate enhanced descriptions if we have analysis results
            if self.analysis_results:
                self.enhance_descriptions()
            else:
                return "No analysis results available for this dataset."
            
            # Double-check that we now have enhanced descriptions
            if not self.enhanced_descriptions:
                return "Unable to generate column descriptions for this dataset."
        
        lines = []
        
        # Title and short dataset summary
        lines.append("# Dataset Column Information")
        
        # Count column types
        categorical_columns = [col for col, info in self.enhanced_descriptions.items() 
                                if info.get("is_categorical", False)]
        numeric_columns = [col for col, info in self.enhanced_descriptions.items() 
                            if info.get("is_numeric", False)]
        datetime_columns = [col for col, info in self.enhanced_descriptions.items() 
                            if info.get("is_datetime", False)]
        
        # Brief dataset overview
        lines.append(f"\nThis dataset contains {len(self.enhanced_descriptions)} columns: "
                    f"{len(categorical_columns)} categorical, {len(numeric_columns)} numeric, and "
                    f"{len(datetime_columns)} datetime columns.")
        
        if self.total_rows > 0:
            lines.append(f"The dataset has approximately {self.total_rows:,} rows.\n")
        
        # Column descriptions - compact format
        lines.append("## Column Descriptions\n")
        
        # First list all columns with types for quick reference
        lines.append("### Column Overview")
        for col_name, info in sorted(self.enhanced_descriptions.items()):
            # Determine column type
            col_type = "Text"
            if info.get("is_id", False): col_type = "ID"
            elif info.get("is_categorical", False): col_type = "Categorical"
            elif info.get("is_numeric", False): col_type = "Numeric"
            elif info.get("is_datetime", False): col_type = "Date/Time"
            
            lines.append(f"- **{col_name}** ({col_type})")
        
        lines.append("\n### Detailed Column Information\n")
        
        # Then provide detailed information for each column
        for col_name, info in sorted(self.enhanced_descriptions.items()):
            # Get column type
            col_type = "Text"
            if info.get("is_id", False): col_type = "ID"
            elif info.get("is_categorical", False): col_type = "Categorical"
            elif info.get("is_numeric", False): col_type = "Numeric"
            elif info.get("is_datetime", False): col_type = "Date/Time"
            
            lines.append(f"**{col_name}** ({col_type}):")
            
            # Add user description if available
            if info.get("has_user_description", False) and info.get("user_description"):
                user_desc = info.get("user_description", "").strip()
                if user_desc:
                    lines.append(f"- Description: {user_desc}")
            
            # Missing values if significant
            missing_pct = info.get("missing_percentage", 0)
            if missing_pct > 1:  # Only mention if above 1%
                lines.append(f"- Missing values: {missing_pct:.1f}%")
            
            # Add type-specific information
            if include_statistics:
                if info.get("is_numeric", False) and "numeric_info" in info:
                    num_info = info["numeric_info"]
                    min_val = self._format_number(num_info.get("min"))
                    max_val = self._format_number(num_info.get("max"))
                    mean_val = self._format_number(num_info.get("mean"))
                    median_val = self._format_number(num_info.get("median"))
                    
                    lines.append(f"- Range: {min_val} to {max_val}")
                    lines.append(f"- Mean: {mean_val}, Median: {median_val}")
                    
                    if num_info.get("has_outliers", False):
                        lines.append("- Contains outliers")
                
                if info.get("is_categorical", False) and "categorical_info" in info:
                    cat_info = info["categorical_info"]
                    unique_count = cat_info.get("unique_count", 0)
                    
                    if unique_count > 0:
                        lines.append(f"- Unique categories: {unique_count}")
                    
                    # Top categories with counts and percentages
                    sorted_top_values = cat_info.get("sorted_top_values", [])
                    if sorted_top_values:
                        top_values_str = []
                        
                        # Limit to specified max categories
                        for value, count in sorted_top_values[:max_categories_per_column]:
                            try:
                                count_numeric = int(count) if isinstance(count, str) else count
                                percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
                                if len(str(value)) > 20:  # Truncate very long values
                                    value_display = f"{str(value)[:17]}..."
                                else:
                                    value_display = str(value)
                                top_values_str.append(f"'{value_display}' ({percentage:.1f}%)")
                            except (ValueError, TypeError):
                                if len(str(value)) > 20:
                                    value_display = f"{str(value)[:17]}..."
                                else:
                                    value_display = str(value)
                                top_values_str.append(f"'{value_display}'")
                        
                        # Only add top categories if we have some
                        if top_values_str:
                            if len(sorted_top_values) > max_categories_per_column:
                                top_cats = ", ".join(top_values_str)
                                lines.append(f"- Top categories: {top_cats}, and {len(sorted_top_values) - max_categories_per_column} more")
                            else:
                                top_cats = ", ".join(top_values_str)
                                lines.append(f"- Top categories: {top_cats}")
                
                if info.get("is_datetime", False) and "datetime_info" in info:
                    date_range = info["datetime_info"].get("date_range", {})
                    if date_range and "min" in date_range and "max" in date_range:
                        lines.append(f"- Date range: {date_range.get('min')} to {date_range.get('max')}")
            
            # Add an empty line between columns
            lines.append("")
        
        # Add tips for querying the dataset
        lines.append("## Suggested Query Approaches")
        lines.append("When asking questions about this dataset, consider:")
        lines.append("- For categorical columns, inquire about distribution and frequencies.")
        lines.append("- For numeric columns, ask about ranges, averages, correlations, and outliers.")
        lines.append("- For datetime columns, explore temporal patterns and trends.")
        lines.append("- Cross-column analysis may reveal interesting relationships in the data.")
        
        return "\n".join(lines)

# class ColumnDescriptionParser:
#     """
#     Parses and enhances column descriptions from various input formats.
#     This class handles user-provided column descriptions and enriches them
#     with insights from data analysis.
#     """
    
#     def __init__(self, col_desc_str=None, analysis_results=None, verbose=True):
#         """
#         Initialize the column description parser.
        
#         Args:
#             col_desc_str (str, optional): Column descriptions as a string
#             analysis_results (Dict[str, Any], optional): Results from DataFrameAnalyzer
#             verbose (bool): Whether to print progress information
#         """
#         self.col_desc_str = col_desc_str or ""
#         self.analysis_results = analysis_results or {}
#         self.verbose = verbose
#         self.col_descriptions = {}
#         self.enhanced_descriptions = {}
#         self.total_rows = 0  # Initialize total_rows with default value
        
#         # Parse descriptions if provided
#         if col_desc_str:
#             self.parse()
            
#         # Enhance descriptions if analysis results provided
#         if analysis_results and self.col_descriptions:
#             self.enhance_descriptions()
    
#     def parse(self, col_desc_str=None) -> Dict[str, str]:
#         """
#         Parse column descriptions from a string format.
#         Handles various input formats including multi-line strings,
#         with column name and description separated by a colon.
        
#         Args:
#             col_desc_str (str, optional): Column descriptions as a string
            
#         Returns:
#             Dict[str, str]: Dictionary of column descriptions
#         """
#         if col_desc_str is not None:
#             self.col_desc_str = col_desc_str
        
#         # Reset descriptions
#         self.col_descriptions = {}
        
#         try:
#             if not self.col_desc_str.strip():
#                 return self.col_descriptions
            
#             # Split the input string into lines and process each line
#             lines = self.col_desc_str.strip().split('\n')
#             current_col = None
#             current_desc = []
            
#             for line in lines:
#                 line = line.strip()
#                 if not line:
#                     continue
                    
#                 # Check if this line starts a new column description
#                 if ':' in line and not current_desc:
#                     parts = line.split(':', 1)
#                     current_col = parts[0].strip()
#                     desc = parts[1].strip()
                    
#                     if current_col and desc:
#                         # This is a single-line description
#                         self.col_descriptions[current_col] = desc
#                         current_col = None
#                     elif current_col:
#                         # This is the start of a multi-line description
#                         current_desc = [desc] if desc else []
#                 else:
#                     # This is a continuation of a multi-line description
#                     if current_col is not None:
#                         current_desc.append(line)
            
#             # Add the last multi-line description if there is one
#             if current_col is not None and current_desc:
#                 self.col_descriptions[current_col] = ' '.join(current_desc)
                
#             if self.verbose:
#                 print(f"Parsed {len(self.col_descriptions)} column descriptions")
                
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error parsing column descriptions: {str(e)}")
        
#         return self.col_descriptions
    
#     def enhance_descriptions(self, analysis_results=None) -> Dict[str, Dict[str, Any]]:
#         """
#         Enhance column descriptions with insights from data analysis.
        
#         Args:
#             analysis_results (Dict[str, Any], optional): Results from DataFrameAnalyzer
            
#         Returns:
#             Dict[str, Dict[str, Any]]: Enhanced column descriptions
#         """
#         if analysis_results is not None:
#             self.analysis_results = analysis_results
        
#         # Reset enhanced descriptions
#         self.enhanced_descriptions = {}
        
#         if not self.analysis_results or not self.col_descriptions:
#             return self.enhanced_descriptions
        
#         try:
#             # Get column details from analysis results
#             column_details = self.analysis_results.get("column_details", {})
            
#             # Get total row count from analysis results
#             self.total_rows = self.analysis_results.get("total_rows", 0)
            
#             # If total_rows is not provided in analysis_results, try to determine it from data
#             if self.total_rows == 0:
#                 # Check if we can find a row count from any column's info
#                 for col_name, col_info in column_details.items():
#                     if "row_count" in col_info:
#                         self.total_rows = col_info["row_count"]
#                         break
            
#             # If still not found, estimate from the first column with top_values
#             if self.total_rows == 0:
#                 for col_name, col_info in column_details.items():
#                     if col_info.get("is_categorical", False) and "top_values" in col_info:
#                         top_values = col_info["top_values"]
#                         total_count = 0
#                         for value, count in top_values.items():
#                             try:
#                                 count_numeric = int(count) if isinstance(count, str) else count
#                                 total_count += count_numeric
#                             except (ValueError, TypeError):
#                                 pass
#                         if total_count > 0:
#                             self.total_rows = total_count
#                             break
            
#             if self.verbose:
#                 if self.total_rows > 0:
#                     print(f"Dataset has {self.total_rows} total rows")
#                 else:
#                     print("Warning: Could not determine total rows, some percentage calculations may be incorrect")
#                     # Set to 1 to avoid division by zero
#                     self.total_rows = 1
            
#             # Process each column with a description
#             for col_name, user_desc in self.col_descriptions.items():
#                 # Start with the user description
#                 enhanced = {
#                     "name": col_name,
#                     "user_description": user_desc,
#                     "has_user_description": True
#                 }
                
#                 # Add analysis insights if available
#                 if col_name in column_details:
#                     col_info = column_details[col_name]
                    
#                     # Add basic column info
#                     enhanced.update({
#                         "type": col_info.get("type", "unknown"),
#                         "is_categorical": col_info.get("is_categorical", False),
#                         "is_numeric": col_info.get("is_numeric", False),
#                         "is_datetime": col_info.get("is_datetime", False),
#                         "is_id": col_info.get("is_id", False),
#                         "missing_count": col_info.get("missing_count", 0),
#                         "missing_percentage": col_info.get("missing_percentage", 0)
#                     })
                    
#                     # Add type-specific information
#                     if col_info.get("is_numeric", False):
#                         enhanced["numeric_info"] = {
#                             "min": col_info.get("min"),
#                             "max": col_info.get("max"),
#                             "mean": col_info.get("mean"),
#                             "median": col_info.get("median"),
#                             "std": col_info.get("std"),
#                             "has_outliers": col_info.get("has_outliers", False)
#                         }
                    
#                     # Enhanced categorical information
#                     if col_info.get("is_categorical", False):
#                         unique_count = col_info.get("unique_count", 0)
#                         top_values = col_info.get("top_values", {})
                        
#                         # Ensure all values in top_values are numeric
#                         top_values_numeric = {}
#                         for k, v in top_values.items():
#                             try:
#                                 top_values_numeric[k] = int(v) if isinstance(v, str) else v
#                             except (ValueError, TypeError):
#                                 top_values_numeric[k] = 0
#                                 if self.verbose:
#                                     print(f"Warning: Could not convert value count '{v}' to int for '{k}' in column {col_name}")
                        
#                         # Create sorted list of top values with numeric counts
#                         sorted_top_values = []
#                         if top_values_numeric:
#                             sorted_top_values = sorted(
#                                 [(k, v) for k, v in top_values_numeric.items()], 
#                                 key=lambda x: x[1], 
#                                 reverse=True
#                             )
                        
#                         # Make sure we have numeric values for the counts
#                         most_common_value = sorted_top_values[0][0] if sorted_top_values else None
#                         most_common_count = int(sorted_top_values[0][1]) if sorted_top_values else 0
                        
#                         # Calculate total and percentages
#                         total_top_values = sum(top_values_numeric.values())
#                         top_value_percentage = float(total_top_values) / self.total_rows * 100 if self.total_rows > 0 else 0
#                         most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 and most_common_count > 0 else 0
                        
#                         enhanced["categorical_info"] = {
#                             "unique_count": unique_count,
#                             "top_values": top_values_numeric,
#                             "sorted_top_values": sorted_top_values,
#                             "top_value_counts": len(top_values),
#                             "top_value_percentage": top_value_percentage,
#                             "most_common": most_common_value,
#                             "most_common_count": most_common_count,
#                             "most_common_percentage": most_common_percentage
#                         }
                    
#                     if col_info.get("is_datetime", False) and "date_range" in col_info:
#                         enhanced["datetime_info"] = {
#                             "date_range": col_info.get("date_range", {})
#                         }
                
#                 # Generate a combined description
#                 combined_desc = self._generate_combined_description(col_name, enhanced)
#                 enhanced["combined_description"] = combined_desc
                
#                 # Add to enhanced descriptions
#                 self.enhanced_descriptions[col_name] = enhanced
            
#             # For columns without user descriptions but with analysis data,
#             # create placeholder enhanced descriptions
#             for col_name, col_info in column_details.items():
#                 if col_name not in self.enhanced_descriptions:
#                     enhanced = {
#                         "name": col_name,
#                         "user_description": "",
#                         "has_user_description": False,
#                         "type": col_info.get("type", "unknown"),
#                         "is_categorical": col_info.get("is_categorical", False),
#                         "is_numeric": col_info.get("is_numeric", False),
#                         "is_datetime": col_info.get("is_datetime", False),
#                         "is_id": col_info.get("is_id", False),
#                         "missing_count": col_info.get("missing_count", 0),
#                         "missing_percentage": col_info.get("missing_percentage", 0)
#                     }
                    
#                     # Add type-specific information
#                     if col_info.get("is_numeric", False):
#                         enhanced["numeric_info"] = {
#                             "min": col_info.get("min"),
#                             "max": col_info.get("max"),
#                             "mean": col_info.get("mean"),
#                             "median": col_info.get("median"),
#                             "std": col_info.get("std"),
#                             "has_outliers": col_info.get("has_outliers", False)
#                         }
                    
#                     # Enhanced categorical information
#                     if col_info.get("is_categorical", False):
#                         unique_count = col_info.get("unique_count", 0)
#                         top_values = col_info.get("top_values", {})
                        
#                         # Ensure all values in top_values are numeric
#                         top_values_numeric = {}
#                         for k, v in top_values.items():
#                             try:
#                                 top_values_numeric[k] = int(v) if isinstance(v, str) else v
#                             except (ValueError, TypeError):
#                                 top_values_numeric[k] = 0
#                                 if self.verbose:
#                                     print(f"Warning: Could not convert value count '{v}' to int for '{k}' in column {col_name}")
                        
#                         # Create sorted list of top values with numeric counts
#                         sorted_top_values = []
#                         if top_values_numeric:
#                             sorted_top_values = sorted(
#                                 [(k, v) for k, v in top_values_numeric.items()], 
#                                 key=lambda x: x[1], 
#                                 reverse=True
#                             )
                        
#                         # Make sure we have numeric values for the counts
#                         most_common_value = sorted_top_values[0][0] if sorted_top_values else None
#                         most_common_count = int(sorted_top_values[0][1]) if sorted_top_values else 0
                        
#                         # Calculate total and percentages
#                         total_top_values = sum(top_values_numeric.values())
#                         top_value_percentage = float(total_top_values) / self.total_rows * 100 if self.total_rows > 0 else 0
#                         most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 and most_common_count > 0 else 0
                        
#                         enhanced["categorical_info"] = {
#                             "unique_count": unique_count,
#                             "top_values": top_values_numeric,
#                             "sorted_top_values": sorted_top_values,
#                             "top_value_counts": len(top_values),
#                             "top_value_percentage": top_value_percentage,
#                             "most_common": most_common_value,
#                             "most_common_count": most_common_count,
#                             "most_common_percentage": most_common_percentage
#                         }
                    
#                     if col_info.get("is_datetime", False) and "date_range" in col_info:
#                         enhanced["datetime_info"] = {
#                             "date_range": col_info.get("date_range", {})
#                         }
                    
#                     # Generate a combined description
#                     combined_desc = self._generate_combined_description(col_name, enhanced)
#                     enhanced["combined_description"] = combined_desc
                    
#                     # Add to enhanced descriptions
#                     self.enhanced_descriptions[col_name] = enhanced
            
#             if self.verbose:
#                 print(f"Enhanced {len(self.enhanced_descriptions)} column descriptions")
                
#         except Exception as e:
#             if self.verbose:
#                 print(f"Error enhancing descriptions: {str(e)}")
#                 import traceback
#                 traceback.print_exc()
        
#         return self.enhanced_descriptions
    
#     def _generate_combined_description(self, col_name: str, enhanced_info: Dict[str, Any]) -> str:
#         """
#         Generate a combined description using user input and analysis insights.
        
#         Args:
#             col_name (str): Name of the column
#             enhanced_info (Dict[str, Any]): Enhanced column information
            
#         Returns:
#             str: Combined description
#         """
#         parts = []
        
#         # Start with user description if available
#         if enhanced_info.get("has_user_description", False):
#             parts.append(enhanced_info["user_description"])
        
#         # Add type information
#         col_type = "unknown"
#         if enhanced_info.get("is_id", False):
#             col_type = "identifier"
#         elif enhanced_info.get("is_datetime", False):
#             col_type = "date/time"
#         elif enhanced_info.get("is_categorical", False):
#             col_type = "categorical"
#         elif enhanced_info.get("is_numeric", False):
#             col_type = "numeric"
#         elif "type" in enhanced_info:
#             col_type = enhanced_info["type"]
        
#         parts.append(f"This is a {col_type} column.")
        
#         # Add missing value information if significant
#         missing_pct = enhanced_info.get("missing_percentage", 0)
#         if missing_pct > 5:  # Only mention if above 5%
#             parts.append(f"It has {missing_pct:.1f}% missing values.")
        
#         # Add type-specific insights
#         if enhanced_info.get("is_numeric", False) and "numeric_info" in enhanced_info:
#             num_info = enhanced_info["numeric_info"]
#             # Format numeric values nicely
#             min_val = self._format_number(num_info.get("min"))
#             max_val = self._format_number(num_info.get("max"))
#             mean_val = self._format_number(num_info.get("mean"))
            
#             parts.append(f"Values range from {min_val} to {max_val}, with an average of {mean_val}.")
            
#             if num_info.get("has_outliers", False):
#                 parts.append("It contains some outlier values.")
        
#         # Enhanced categorical information with more details
#         if enhanced_info.get("is_categorical", False) and "categorical_info" in enhanced_info:
#             cat_info = enhanced_info["categorical_info"]
#             unique_count = cat_info.get("unique_count", 0)
            
#             if unique_count > 0:
#                 parts.append(f"It contains {unique_count} unique categories.")
            
#             # Include most common value and its frequency
#             most_common = cat_info.get("most_common")
#             most_common_pct = cat_info.get("most_common_percentage", 0)
#             if most_common:
#                 parts.append(f"The most common value is '{most_common}', appearing in approximately {most_common_pct:.1f}% of the data.")
            
#             # Include all top categories with counts and percentages
#             sorted_top_values = cat_info.get("sorted_top_values", [])
#             if len(sorted_top_values) > 1:  # If there are multiple top values
#                 top_values_str = []
#                 for value, count in sorted_top_values[:5]:  # Limit to top 5 for readability
#                     # Ensure count is numeric
#                     try:
#                         count_numeric = int(count) if isinstance(count, str) else count
#                         percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
#                         top_values_str.append(f"'{value}' ({count_numeric} occurrences, {percentage:.1f}%)")
#                     except (ValueError, TypeError):
#                         top_values_str.append(f"'{value}' ({count} occurrences)")
                
#                 if len(sorted_top_values) > 5:
#                     top_values_list = ", ".join(top_values_str)
#                     parts.append(f"Top categories include: {top_values_list}, and {len(sorted_top_values) - 5} more.")
#                 else:
#                     top_values_list = ", ".join(top_values_str)
#                     parts.append(f"Top categories include: {top_values_list}.")
        
#         if enhanced_info.get("is_datetime", False) and "datetime_info" in enhanced_info:
#             date_range = enhanced_info["datetime_info"].get("date_range", {})
#             if "min" in date_range and "max" in date_range:
#                 parts.append(f"Dates span from {date_range['min']} to {date_range['max']}.")
        
#         return " ".join(parts)
    
#     def _format_number(self, value) -> str:
#         """
#         Format a number for display in descriptions.
        
#         Args:
#             value: Numeric value to format
            
#         Returns:
#             str: Formatted number
#         """
#         if value is None:
#             return "unknown"
        
#         try:
#             # Integer formatting
#             if value == int(value):
#                 return f"{int(value):,}"
            
#             # Float formatting - different precision based on magnitude
#             abs_val = abs(value)
#             if abs_val >= 1000:
#                 return f"{value:,.0f}"
#             elif abs_val >= 100:
#                 return f"{value:,.1f}"
#             elif abs_val >= 1:
#                 return f"{value:,.2f}"
#             elif abs_val == 0:
#                 return "0"
#             else:
#                 # For very small numbers, use scientific notation
#                 return f"{value:.2e}"
#         except:
#             return str(value)
    
#     def get_descriptions_dict(self) -> Dict[str, str]:
#         """
#         Get the parsed column descriptions as a simple dictionary.
        
#         Returns:
#             Dict[str, str]: Dictionary mapping column names to descriptions
#         """
#         return self.col_descriptions
    
#     def get_enhanced_descriptions(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Get the enhanced column descriptions.
        
#         Returns:
#             Dict[str, Dict[str, Any]]: Enhanced column descriptions
#         """
#         return self.enhanced_descriptions
    
#     def get_combined_descriptions(self) -> Dict[str, str]:
#         """
#         Get the combined descriptions (user + analysis insights).
        
#         Returns:
#             Dict[str, str]: Dictionary mapping column names to combined descriptions
#         """
#         combined = {}
#         for col_name, info in self.enhanced_descriptions.items():
#             combined[col_name] = info.get("combined_description", "")
#         return combined
    
#     def print_descriptions(self):
#         """
#         Print the parsed column descriptions.
#         """
#         if not self.col_descriptions:
#             print("No column descriptions available.")
#             return
            
#         print("\n" + "=" * 50)
#         print("COLUMN DESCRIPTIONS")
#         print("=" * 50)
        
#         for col, desc in sorted(self.col_descriptions.items()):
#             print(f"\n{col}:")
#             print(f"  {desc}")
    
#     def print_enhanced_descriptions(self):
#         """
#         Print the enhanced column descriptions.
#         """
#         if not self.enhanced_descriptions:
#             print("No enhanced descriptions available.")
#             return
            
#         print("\n" + "=" * 50)
#         print("ENHANCED COLUMN DESCRIPTIONS")
#         print("=" * 50)
        
#         for col, info in sorted(self.enhanced_descriptions.items()):
#             print(f"\n{col}:")
#             if info.get("has_user_description", False):
#                 print(f"  User Description: {info['user_description']}")
#             print(f"  Combined Description: {info['combined_description']}")
            
#             # Print type information
#             col_type = []
#             if info.get("is_id", False): col_type.append("ID")
#             if info.get("is_categorical", False): col_type.append("Categorical")
#             if info.get("is_numeric", False): col_type.append("Numeric")
#             if info.get("is_datetime", False): col_type.append("Date/Time")
            
#             print(f"  Type: {', '.join(col_type) if col_type else info.get('type', 'Unknown')}")
            
#             # Print statistics for numeric columns
#             if info.get("is_numeric", False) and "numeric_info" in info:
#                 num_info = info["numeric_info"]
#                 print(f"  Range: {self._format_number(num_info.get('min'))} to {self._format_number(num_info.get('max'))}")
#                 print(f"  Mean: {self._format_number(num_info.get('mean'))}, Median: {self._format_number(num_info.get('median'))}")
            
#             # Enhanced categorical information display
#             if info.get("is_categorical", False) and "categorical_info" in info:
#                 cat_info = info["categorical_info"]
#                 print(f"  Unique Categories: {cat_info.get('unique_count', 0)}")
                
#                 # Show most common value
#                 if cat_info.get("most_common") is not None:
#                     try:
#                         most_common_count = cat_info.get("most_common_count", 0)
#                         # Ensure most_common_count is numeric
#                         if isinstance(most_common_count, str):
#                             most_common_count = int(most_common_count)
#                         print(f"  Most Common: '{cat_info.get('most_common')}' ({most_common_count} occurrences, {cat_info.get('most_common_percentage', 0):.1f}%)")
#                     except (ValueError, TypeError):
#                         print(f"  Most Common: '{cat_info.get('most_common')}' (error calculating stats)")
                
#                 # Show all top values with counts and percentages
#                 print("  Top Categories:")
#                 sorted_top_values = cat_info.get("sorted_top_values", [])
#                 for i, (value, count) in enumerate(sorted_top_values[:10]):  # Show up to 10 top values
#                     try:
#                         # Ensure count is numeric
#                         count_numeric = int(count) if isinstance(count, str) else count
#                         percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
#                         print(f"    {i+1}. '{value}': {count_numeric} occurrences ({percentage:.1f}%)")
#                     except (ValueError, TypeError):
#                         print(f"    {i+1}. '{value}': {count} occurrences (error calculating percentage)")
                
#                 if len(sorted_top_values) > 10:
#                     print(f"    ... and {len(sorted_top_values) - 10} more categories")
            
#             # Print date range for datetime columns
#             if info.get("is_datetime", False) and "datetime_info" in info:
#                 date_range = info["datetime_info"].get("date_range", {})
#                 if date_range:
#                     print(f"  Date Range: {date_range.get('min', 'Unknown')} to {date_range.get('max', 'Unknown')}")
    
#     def get_categorical_summary(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Get a summary of all categorical columns with detailed category information.
        
#         Returns:
#             Dict[str, Dict[str, Any]]: Summary of categorical columns
#         """
#         categorical_summary = {}
        
#         # Check if total_rows is set, if not try to get it from analysis_results
#         if self.total_rows == 0 and self.analysis_results:
#             self.total_rows = self.analysis_results.get("total_rows", 0)
#             if self.verbose and self.total_rows > 0:
#                 print(f"Setting total_rows to {self.total_rows} in get_categorical_summary")
        
#         # If still zero, try to calculate from the data
#         if self.total_rows == 0:
#             # Try to estimate total rows from the first categorical column's data
#             for col_info in self.enhanced_descriptions.values():
#                 if col_info.get("is_categorical", False) and "categorical_info" in col_info:
#                     cat_info = col_info["categorical_info"]
#                     sorted_top_values = cat_info.get("sorted_top_values", [])
#                     if sorted_top_values:
#                         total_count = 0
#                         for _, count in sorted_top_values:
#                             try:
#                                 count_numeric = int(count) if isinstance(count, str) else count
#                                 total_count += count_numeric
#                             except (ValueError, TypeError):
#                                 pass
#                         if total_count > 0:
#                             self.total_rows = total_count
#                             if self.verbose:
#                                 print(f"Estimated total_rows as {self.total_rows} from category counts")
#                             break
        
#         # Default to 1 to avoid division by zero if we couldn't determine total_rows
#         if self.total_rows == 0:
#             self.total_rows = 1
#             if self.verbose:
#                 print("Warning: Could not determine total_rows, using default value of 1")
                
#         for col_name, info in self.enhanced_descriptions.items():
#             if info.get("is_categorical", False) and "categorical_info" in info:
#                 cat_info = info["categorical_info"]
                
#                 # Make sure most_common_count is numeric
#                 most_common_count = 0
#                 try:
#                     most_common_count_val = cat_info.get("most_common_count", 0)
#                     most_common_count = int(most_common_count_val) if isinstance(most_common_count_val, str) else most_common_count_val
#                 except (ValueError, TypeError):
#                     if self.verbose:
#                         print(f"Warning: Could not convert most_common_count to int for column {col_name}")
                
#                 # Calculate most_common_percentage
#                 most_common_percentage = 0
#                 try:
#                     most_common_percentage = float(most_common_count) / self.total_rows * 100 if self.total_rows > 0 else 0
#                 except (ValueError, TypeError):
#                     if self.verbose:
#                         print(f"Warning: Error calculating most_common_percentage for column {col_name}")
                
#                 summary = {
#                     "column_name": col_name,
#                     "description": info.get("user_description", ""),
#                     "combined_description": info.get("combined_description", ""),
#                     "unique_categories": cat_info.get("unique_count", 0),
#                     "most_common": cat_info.get("most_common"),
#                     "most_common_count": most_common_count,
#                     "most_common_percentage": most_common_percentage,
#                     "categories": []
#                 }
                
#                 # Add all categories with their counts and percentages
#                 for value, count in cat_info.get("sorted_top_values", []):
#                     try:
#                         # Ensure count is numeric
#                         count_numeric = int(count) if isinstance(count, str) else count
#                         percentage = float(count_numeric) / self.total_rows * 100 if self.total_rows > 0 else 0
#                         summary["categories"].append({
#                             "value": value,
#                             "count": count_numeric,
#                             "percentage": percentage
#                         })
#                     except (ValueError, TypeError):
#                         if self.verbose:
#                             print(f"Warning: Error processing category '{value}' for column {col_name}")
                
#                 categorical_summary[col_name] = summary
        
#         return categorical_summary

#     def get_output_for_next_stage(self) -> Dict[str, Any]:
#         """
#         Prepare a comprehensive output package for the next processing stage.
#         This method returns a carefully organized dictionary with all column descriptions,
#         interpretations, and metadata, formatted for easy consumption by downstream components.
        
#         Returns:
#             Dict[str, Any]: Structured output for next stage
#         """
#         # Make sure we have enhanced descriptions
#         if not self.enhanced_descriptions:
#             # Try to generate enhanced descriptions if we have analysis results
#             if self.analysis_results:
#                 self.enhance_descriptions()
#             else:
#                 if self.verbose:
#                     print("No analysis results available. Cannot generate enhanced descriptions.")
#                 return {}
            
#             # Double-check that we now have enhanced descriptions
#             if not self.enhanced_descriptions:
#                 if self.verbose:
#                     print("Unable to generate enhanced descriptions.")
#                 return {}
        
#         output_package = {
#             "metadata": {
#                 "total_columns": len(self.enhanced_descriptions),
#                 "columns_with_user_descriptions": sum(1 for info in self.enhanced_descriptions.values() 
#                                                     if info.get("has_user_description", False)),
#                 "categorical_columns": sum(1 for info in self.enhanced_descriptions.values() 
#                                             if info.get("is_categorical", False)),
#                 "numeric_columns": sum(1 for info in self.enhanced_descriptions.values() 
#                                         if info.get("is_numeric", False)),
#                 "datetime_columns": sum(1 for info in self.enhanced_descriptions.values() 
#                                         if info.get("is_datetime", False)),
#                 "id_columns": sum(1 for info in self.enhanced_descriptions.values() 
#                                 if info.get("is_id", False))
#             },
#             "columns": {},
#             "text_descriptions": {},
#             "column_groups": {
#                 "categorical": [],
#                 "numeric": [],
#                 "datetime": [],
#                 "id": [],
#                 "text": [],
#                 "with_user_description": [],
#                 "without_user_description": []
#             }
#         }
        
#         # Process each column
#         for col_name, info in self.enhanced_descriptions.items():
#             # Add to column groups
#             if info.get("is_categorical", False):
#                 output_package["column_groups"]["categorical"].append(col_name)
#             if info.get("is_numeric", False):
#                 output_package["column_groups"]["numeric"].append(col_name)
#             if info.get("is_datetime", False):
#                 output_package["column_groups"]["datetime"].append(col_name)
#             if info.get("is_id", False):
#                 output_package["column_groups"]["id"].append(col_name)
#             if not any([info.get("is_categorical", False), 
#                         info.get("is_numeric", False),
#                         info.get("is_datetime", False),
#                         info.get("is_id", False)]):
#                 output_package["column_groups"]["text"].append(col_name)
            
#             if info.get("has_user_description", False):
#                 output_package["column_groups"]["with_user_description"].append(col_name)
#             else:
#                 output_package["column_groups"]["without_user_description"].append(col_name)
            
#             # Add text descriptions
#             output_package["text_descriptions"][col_name] = {
#                 "user_description": info.get("user_description", ""),
#                 "combined_description": info.get("combined_description", "")
#             }
            
#             # Add structured column information
#             column_data = {
#                 "name": col_name,
#                 "has_user_description": info.get("has_user_description", False),
#                 "type": info.get("type", "unknown"),
#                 "is_categorical": info.get("is_categorical", False),
#                 "is_numeric": info.get("is_numeric", False),
#                 "is_datetime": info.get("is_datetime", False),
#                 "is_id": info.get("is_id", False),
#                 "missing_count": info.get("missing_count", 0),
#                 "missing_percentage": info.get("missing_percentage", 0)
#             }
            
#             # Add type-specific information
#             if info.get("is_numeric", False) and "numeric_info" in info:
#                 column_data["numeric_stats"] = info["numeric_info"]
            
#             if info.get("is_categorical", False) and "categorical_info" in info:
#                 # Ensure counts are numeric
#                 categories = []
#                 try:
#                     for val, count in info["categorical_info"].get("sorted_top_values", []):
#                         try:
#                             count_numeric = int(count) if isinstance(count, str) else count
#                             percentage = count_numeric / self.total_rows * 100 if self.total_rows > 0 else 0
#                             categories.append({
#                                 "value": val,
#                                 "count": count_numeric,
#                                 "percentage": percentage
#                             })
#                         except (ValueError, TypeError):
#                             if self.verbose:
#                                 print(f"Warning: Error processing category '{val}' in get_output_for_next_stage")
                    
#                     most_common_count = info["categorical_info"].get("most_common_count", 0)
#                     if isinstance(most_common_count, str):
#                         most_common_count = int(most_common_count)
                    
#                     column_data["categorical_stats"] = {
#                         "unique_count": info["categorical_info"].get("unique_count", 0),
#                         "most_common": info["categorical_info"].get("most_common"),
#                         "most_common_count": most_common_count,
#                         "most_common_percentage": info["categorical_info"].get("most_common_percentage", 0),
#                         "categories": categories
#                     }
#                 except Exception as e:
#                     if self.verbose:
#                         print(f"Error creating categorical stats for {col_name}: {str(e)}")
#                     # Provide a minimal version of categorical stats to avoid failures
#                     column_data["categorical_stats"] = {
#                         "unique_count": info["categorical_info"].get("unique_count", 0),
#                         "most_common": info["categorical_info"].get("most_common"),
#                         "most_common_count": 0,
#                         "most_common_percentage": 0,
#                         "categories": []
#                     }
            
#             if info.get("is_datetime", False) and "datetime_info" in info:
#                 column_data["datetime_stats"] = info["datetime_info"]
            
#             output_package["columns"][col_name] = column_data
        
#         return output_package

def generate_dataset_report_for_llm(
    df, 
    col_desc_str, 
    openai_api_key=None, 
    model_name="gpt-3.5-turbo", 
    verbose=True
):
    """
    Generate an LLM-friendly report for a dataset with formatted column descriptions.
    
    Args:
        df: pandas DataFrame to analyze
        col_desc_str: Raw string with column descriptions
        openai_api_key: OpenAI API key for formatting descriptions (optional if already set)
        model_name: Model to use for formatting descriptions
        verbose: Whether to print progress information
    
    Returns:
        str: Markdown-formatted dataset report optimized for LLM context
    """
    
    # Step 1: Create analyzer and run analysis
    analyzer = DataFrameAnalyzer(df, verbose=verbose)
    basic_info = analyzer.analyze()
    analysis_results = analyzer.get_analysis_for_next_stage()
    
    # Step 2: Format column descriptions using LLM
    prompt_template = PromptTemplate(
        input_variables=["raw_text", "columns"],
        template="""
        You are a helpful assistant. Your task is to extract and format column descriptions.

        Given:
        1. A list of column names from a DataFrame.
        2. A block of unstructured text describing the columns.

        Instructions:
        - Match each column name as accurately as possible using fuzzy matching.
        - Output each matched column description in the following format:
            column_name: description
        - Use only column names from the provided list.

        Column names:
        {columns}

        Raw description:
        \"\"\"
        {raw_text}
        \"\"\"

        Formatted output:
        """)
    
    # Define LLM with provided API key
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=openai_api_key)
    
    # Create LLMChain
    chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=verbose
    )
    
    # Run chain with actual inputs
    formatted_col_desc = chain.run({
        "raw_text": col_desc_str,
        "columns": df.columns.tolist()
    })
    
    # Step 3: Create parser with formatted descriptions and analysis results
    parser = ColumnDescriptionParser(formatted_col_desc, analysis_results, verbose=verbose)
    
    # Step 4: Generate LLM-friendly report
    llm_report = parser.get_llm_context_report()
    
    return llm_report