# modules/data_preparation.py

import pandas as pd
import numpy as np
import re
from datetime import datetime

def load_dataframe(csv_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        DataFrame: Pandas DataFrame containing the CSV data.
    """
    try:
        df = pd.read_csv(csv_path)
        print(f"\n✅ Loaded '{csv_path}' with shape {df.shape}.")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def load_column_descriptions(description_path):
    """
    Load column descriptions from a file.
    
    Args:
        description_path (str): Path to the file containing column descriptions.
        
    Returns:
        str: String containing column descriptions.
    """
    try:
        with open(description_path, "r") as f:
            col_desc = f.read()
        return col_desc
    except Exception as e:
        print(f"❌ Error loading column descriptions: {e}")
        return ""

def prepare_dataframe_globals():
    """
    Prepare the global variables needed for the Python tool.
    
    Returns:
        dict: Dictionary containing global variables for the Python tool.
    """
    return {
        "pd": pd,
        "np": np,
    }

def get_dataframe_info(df):
    """
    Get information about the DataFrame.
    
    Args:
        df (DataFrame): Pandas DataFrame.
        
    Returns:
        str: String representation of DataFrame info.
    """
    # Create a string buffer to capture the output of df.info()
    import io
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()




class DataFramePreprocessor:
    """
    A utility class to automatically preprocess pandas DataFrames by intelligently
    detecting and transforming column types.
    """
    
    def __init__(self, df=None):
        """
        Initialize the preprocessor with an optional DataFrame.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to preprocess
        """
        self.df = df.copy() if df is not None else None
        self.transformations_applied = {}
    
    def fit_transform(self, df=None):
        """
        Process the DataFrame by applying appropriate transformations to each column.
        
        Args:
            df (pd.DataFrame, optional): DataFrame to preprocess, uses the one provided in initialization if None
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        if df is not None:
            self.df = df.copy()
            
        if self.df is None:
            raise ValueError("No DataFrame provided for preprocessing")
            
        # Reset transformations applied
        self.transformations_applied = {}
        
        # Process each column
        for column in self.df.columns:
            self._process_column(column)
            
        return self.df
    
    def _process_column(self, column):
        """
        Process a single column by detecting its type and applying appropriate transformations.
        
        Args:
            column (str): Column name to process
        """
        series = self.df[column]
        
        # Skip processing if the column is entirely null
        if series.isna().all():
            self.transformations_applied[column] = ["skipped - all null values"]
            return
        
        transformations = []
        
        # Clean column name
        clean_name = self._clean_column_name(column)
        if False : #& clean_name != column 
            self.df.rename(columns={column: clean_name}, inplace=True)
            column = clean_name
            transformations.append("renamed column")
        
        # Handle string processing for object types
        if series.dtype == 'object':
            # Try to convert to datetime first
            if self._is_likely_datetime(series):
                self.df[column] = self._convert_to_datetime(series)
                transformations.append("converted to datetime")
            else:
                # String cleaning operations
                self.df[column] = series.apply(lambda x: x.strip() if isinstance(x, str) else x)
                transformations.append("trimmed whitespace")
                
                # Check for boolean columns
                if self._is_likely_boolean(series):
                    self.df[column] = self._convert_to_boolean(series)
                    transformations.append("converted to boolean")
                # Check for numeric columns with wrong type
                elif self._is_likely_numeric(series):
                    self.df[column] = self._convert_to_numeric(series)
                    transformations.append("converted to numeric")
                else:
                    # If still object type, categorize if possible
                    unique_ratio = series.nunique() / len(series)
                    if unique_ratio < 0.5 and series.nunique() < 100:  # Heuristic for categorical
                        self.df[column] = self.df[column].astype('category')
                        transformations.append("converted to category")
        
        # Handle numeric types
        elif pd.api.types.is_numeric_dtype(series):
            # Check for integers stored as floats
            if series.dtype == 'float64' and series.dropna().apply(lambda x: x.is_integer()).all():
                self.df[column] = series.astype(pd.Int64Dtype())  # Pandas nullable integer
                transformations.append("converted float to integer (nullable)")
        
        # Record transformations
        self.transformations_applied[column] = transformations if transformations else ["no transformations needed"]
    
    def _clean_column_name(self, column):
        """
        Clean a column name by replacing spaces with underscores and lowercasing.
        
        Args:
            column (str): Column name to clean
            
        Returns:
            str: Cleaned column name
        """
        # Replace spaces and special characters with underscores
        cleaned = re.sub(r'[^\w\s]', '_', column)
        cleaned = re.sub(r'\s+', '_', cleaned)
        # Remove duplicate underscores and ensure lowercase
        cleaned = re.sub(r'_+', '_', cleaned).lower().strip('_')
        return cleaned
    
    def _is_likely_datetime(self, series):
        """
        Detect if a series likely contains datetime values.
        
        Args:
            series (pd.Series): Series to check
            
        Returns:
            bool: True if likely datetime, False otherwise
        """
        # Common date patterns to check
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',     # YYYY/MM/DD
            r'\d{1,2}-[A-Za-z]{3}-\d{2,4}',     # DD-MMM-YYYY
            r'[A-Za-z]{3,9} \d{1,2},? \d{4}',   # Month DD, YYYY
            r'\d{1,2} [A-Za-z]{3,9},? \d{4}'    # DD Month YYYY
        ]
        
        # Check for datetime patterns in a sample of non-null values
        non_null = series.dropna()
        sample_size = min(100, len(non_null))
        sample = non_null.sample(sample_size) if sample_size > 0 else non_null
        
        for pattern in date_patterns:
            matched = sample.astype(str).str.match(pattern)
            if matched.sum() / sample_size > 0.7:  # If >70% match pattern
                return True
        
        # Check for ISO format dates/times
        try:
            pd.to_datetime(sample)
            return True
        except:
            return False
    
    def _convert_to_datetime(self, series):
        """
        Convert a series to datetime format.
        
        Args:
            series (pd.Series): Series to convert
            
        Returns:
            pd.Series: Converted series
        """
        try:
            return pd.to_datetime(series, errors='coerce')
        except:
            # If conversion fails, return original series
            return series
    
    def _is_likely_boolean(self, series):
        """
        Detect if a series likely contains boolean values.
        
        Args:
            series (pd.Series): Series to check
            
        Returns:
            bool: True if likely boolean, False otherwise
        """
        # Common boolean values
        boolean_values = {
            'true', 'false', 't', 'f', 'yes', 'no', 'y', 'n', '1', '0', 
            'True', 'False', 'TRUE', 'FALSE', 'Yes', 'No', 'Y', 'N'
        }
        
        unique_values = set(series.dropna().astype(str).unique())
        return unique_values.issubset(boolean_values)
    
    def _convert_to_boolean(self, series):
        """
        Convert a series to boolean format.
        
        Args:
            series (pd.Series): Series to convert
            
        Returns:
            pd.Series: Converted series
        """
        # Map values to boolean
        true_values = {'true', 't', 'yes', 'y', '1', 'True', 'TRUE', 'Yes', 'Y'}
        
        def map_to_bool(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, str):
                return True if x.lower() in true_values else False
            return bool(x)
        
        return series.apply(map_to_bool)
    
    def _is_likely_numeric(self, series):
        """
        Detect if a series of strings likely contains numeric values.
        
        Args:
            series (pd.Series): Series to check
            
        Returns:
            bool: True if likely numeric, False otherwise
        """
        # Check if values can be converted to numeric
        non_null = series.dropna()
        numeric_conversion = pd.to_numeric(non_null, errors='coerce')
        
        # Consider it numeric if >70% can be converted
        return numeric_conversion.notna().sum() / len(non_null) > 0.7
    
    def _convert_to_numeric(self, series):
        """
        Convert a series to numeric format, handling currency symbols and commas.
        
        Args:
            series (pd.Series): Series to convert
            
        Returns:
            pd.Series: Converted series
        """
        # Pre-process strings to remove currency symbols and commas
        def clean_numeric_string(x):
            if not isinstance(x, str):
                return x
            # Remove currency symbols, commas, and other non-numeric characters except decimal points
            clean_str = re.sub(r'[^\d.-]', '', x)
            return clean_str
        
        clean_series = series.apply(clean_numeric_string)
        return pd.to_numeric(clean_series, errors='coerce')
    
    def get_summary(self):
        """
        Get a summary of transformations applied.
        
        Returns:
            dict: Dictionary of transformations applied to each column
        """
        return self.transformations_applied
    


def prepare_dataframe(csv_path, description_path=None):
    """
    Main function to prepare the dataframe and related information.
    
    Args:
        csv_path (str): Path to the CSV file.
        description_path (str, optional): Path to column descriptions file.
        
    Returns:
        tuple: (DataFrame, DataFrame info string, column descriptions string, globals dict)
    """
    # Load the dataframe
    df = load_dataframe(csv_path)

    # Show results
    print("Original DataFrame:")
    print(df.dtypes)
    
    ### Add function to make sure that dataframe if preprocessed
    preprocessor = DataFramePreprocessor()
    df = preprocessor.fit_transform(df)
    

    print("\nProcessed DataFrame:")
    print(df.dtypes)
    print("\nTransformations applied:")
    for col, transforms in preprocessor.get_summary().items():
        print(f"{col}: {', '.join(transforms)}")
    
    
    # Create globals dictionary and add dataframe
    globals_dict = prepare_dataframe_globals()
    globals_dict["df"] = df
    
    # Get dataframe info
    df_info_str = get_dataframe_info(df)
    
    # Load column descriptions if provided
    col_desc_str = ""
    if description_path:
        col_desc_str = load_column_descriptions(description_path)
    
    return df, df_info_str, col_desc_str, globals_dict
