# modules/data_preparation.py

import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
import json
from typing import List, Dict, Any, Union, Optional, Tuple
import openai

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

class AIDataPreprocessor:
    """
    An intelligent data preprocessing agent that uses OpenAI's GPT models to automatically
    analyze DataFrame columns and determine appropriate transformations.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo"):
        """
        Initialize the AI-powered preprocessor.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, will try to use OPENAI_API_KEY environment variable
            model (str): OpenAI model to use for analysis. Default is gpt-4-turbo.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as parameter or as OPENAI_API_KEY environment variable")
        
        self.model = model
        self.client = openai.OpenAI(api_key=self.api_key)
        self.transformations_applied = {}
        self.transformation_code = {}
        self.df = None
    
    def fit_transform(self, df: pd.DataFrame, sample_size: int = 100) -> pd.DataFrame:
        """
        Process the DataFrame using AI to determine appropriate transformations.
        
        Args:
            df (pd.DataFrame): DataFrame to preprocess
            sample_size (int): Number of samples to analyze for each column (to limit API usage)
            
        Returns:
            pd.DataFrame: Preprocessed DataFrame
        """
        self.df = df.copy()
        self.transformations_applied = {}
        
        # Process columns in batches to reduce API calls
        columns = list(df.columns)
        results = []
        
        for column in columns:
            column_data = self._prepare_column_sample(column, sample_size)
            transformations = self._analyze_column(column, column_data)
            results.append((column, transformations))
        
        # Apply transformations
        for column, transformations in results:
            self._apply_transformations(column, transformations)
        
        return self.df
    
    def _prepare_column_sample(self, column: str, sample_size: int) -> Dict[str, Any]:
        """
        Prepare a sample of column data for AI analysis.
        
        Args:
            column (str): Column name
            sample_size (int): Number of samples to include
            
        Returns:
            Dict: Dictionary containing column metadata and samples
        """
        series = self.df[column]
        
        # Get non-null values for sampling
        non_null = series.dropna()
        
        # Calculate sample size (min of specified or available data)
        actual_sample_size = min(sample_size, len(non_null))
        
        # Get a representative sample
        if actual_sample_size > 0:
            sample = non_null.sample(actual_sample_size) if len(non_null) > actual_sample_size else non_null
        else:
            sample = pd.Series([])
        
        # Create a column info dictionary
        column_info = {
            "name": column,
            "dtype": str(series.dtype),
            "unique_count": series.nunique(),
            "null_count": series.isna().sum(),
            "total_count": len(series),
            "sample_values": sample.tolist() if not sample.empty else [],
            "unique_values": series.unique().tolist()[:20] if series.nunique() <= 20 else None
        }
        
        return column_info
    
    def _analyze_column(self, column: str, column_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use OpenAI to analyze the column and determine appropriate transformations.
        
        Args:
            column (str): Column name
            column_data (Dict): Column metadata and samples
            
        Returns:
            Dict: Dictionary of recommended transformations
        """
        prompt = self._create_analysis_prompt(column_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data preprocessing expert. Analyze the column data and suggest appropriate transformations."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            transformations = json.loads(response.choices[0].message.content)
            return transformations
        
        except Exception as e:
            print(f"Error analyzing column {column}: {str(e)}")
            # Return a default no-transformation result
            return {
                "column_type": "unknown",
                "recommended_transformations": [],
                "transformation_code": "# No transformations due to API error\ndf[column_name] = df[column_name]"
            }
    
    def _create_analysis_prompt(self, column_data: Dict[str, Any]) -> str:
        """
        Create a prompt for the AI to analyze the column.
        
        Args:
            column_data (Dict): Column metadata and samples
            
        Returns:
            str: Prompt for AI analysis
        """
        prompt = f"""
            Analyze the following pandas DataFrame column and recommend appropriate preprocessing transformations.

            Column Name: {column_data['name']}
            Current Data Type: {column_data['dtype']}
            Unique Values Count: {column_data['unique_count']}
            Null Values Count: {column_data['null_count']}
            Total Rows: {column_data['total_count']}

            Sample Values:
            {json.dumps(column_data['sample_values'][:10], indent=2)}

            {"Unique Values (Complete List):" + json.dumps(column_data['unique_values'], indent=2) if column_data['unique_values'] is not None else "Too many unique values to list"}

            Determine the most appropriate data type and transformations for this column. Consider:
            1. Does it look like dates/times that should be datetime?
            2. Are there strings that need cleaning (whitespace trimming, case normalization)?
            3. Is it categorical data that should be converted to category dtype?
            4. Is it numeric data with formatting (currency symbols, commas) that needs cleaning?
            5. Are there boolean values represented as strings ('Yes'/'No', 'True'/'False')?
            6. Does the column name need cleaning (spaces, special chars)?

            Provide your analysis as a JSON object with the following structure:
            ```json
            {
            "column_type": "string|numeric|datetime|boolean|categorical|other",
            "recommended_transformations": ["list", "of", "transformations"],
            "transformation_code": "Python code using pandas to transform this column (use 'column_name' as placeholder)"
            }
            ```

            Focus on practical transformations that will improve data quality and usability.
            """
        return prompt
    
    def _apply_transformations(self, column: str, transformations: Dict[str, Any]) -> None:
        """
        Apply the AI-recommended transformations to the column.
        
        Args:
            column (str): Column name
            transformations (Dict): Transformation recommendations from AI
        """
        # Store the transformations for later reference
        self.transformations_applied[column] = transformations.get("recommended_transformations", [])
        self.transformation_code[column] = transformations.get("transformation_code", "")
        
        # Execute the transformation code
        if transformations.get("transformation_code"):
            try:
                # Create a safe execution environment
                local_vars = {"df": self.df, "column_name": column, "pd": pd, "np": np, "re": re}
                
                # Execute the transformation code
                exec(transformations["transformation_code"], {"__builtins__": {}}, local_vars)
                
                # Update the DataFrame (in case it was replaced in the local scope)
                self.df = local_vars["df"]
            except Exception as e:
                print(f"Error applying transformation to {column}: {str(e)}")
                # Log the error in transformations
                self.transformations_applied[column].append(f"ERROR: {str(e)}")
    
    def get_summary(self) -> Dict[str, List[str]]:
        """
        Get a summary of transformations applied.
        
        Returns:
            Dict: Dictionary of transformations applied to each column
        """
        return self.transformations_applied
    
    def get_transformation_code(self) -> Dict[str, str]:
        """
        Get the actual code used for transformations.
        
        Returns:
            Dict: Dictionary of transformation code for each column
        """
        return self.transformation_code

# Example usage with dry run option (no API calls)
class DryRunAIDataPreprocessor(AIDataPreprocessor):
    """
    A version of AIDataPreprocessor that doesn't make actual API calls.
    Useful for testing and demonstration.
    """
    
    def __init__(self):
        """Initialize without requiring an API key"""
        self.transformations_applied = {}
        self.transformation_code = {}
        self.df = None
    
    def _analyze_column(self, column: str, column_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate AI analysis based on column characteristics.
        
        Args:
            column (str): Column name
            column_data (Dict): Column metadata and samples
            
        Returns:
            Dict: Dictionary of recommended transformations
        """
        sample_values = column_data['sample_values']
        dtype = column_data['dtype']
        name = column_data['name']
        unique_count = column_data['unique_count']
        total_count = column_data['total_count']
        
        # Simulate AI analysis based on basic patterns
        transformations = {
            "column_type": "unknown",
            "recommended_transformations": [],
            "transformation_code": f"# No transformations needed\ndf[column_name] = df[column_name]"
        }
        
        # Check for column name cleaning needs
        if re.search(r'[^\w]', name) or not name.islower():
            clean_name = re.sub(r'[^\w\s]', '_', name)
            clean_name = re.sub(r'\s+', '_', clean_name).lower().strip('_')
            if clean_name != name:
                transformations["recommended_transformations"].append("clean column name")
                transformations["transformation_code"] = f"# Clean column name\ndf.rename(columns={{column_name: '{clean_name}'}}, inplace=True)"
                return transformations
        
        # Simple date detection
        if len(sample_values) > 0 and all(isinstance(x, str) for x in sample_values):
            date_patterns = [
                r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY, DD/MM/YYYY
                r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',     # YYYY/MM/DD
                r'[A-Za-z]{3,9} \d{1,2},? \d{4}',   # Month DD, YYYY
            ]
            
            is_date = False
            for pattern in date_patterns:
                if any(re.match(pattern, str(x)) for x in sample_values if isinstance(x, str)):
                    is_date = True
                    break
                    
            if is_date:
                transformations["column_type"] = "datetime"
                transformations["recommended_transformations"] = ["convert to datetime"]
                transformations["transformation_code"] = "# Convert to datetime\ndf[column_name] = pd.to_datetime(df[column_name], errors='coerce')"
                return transformations
        
        # Check for boolean values
        if len(sample_values) > 0:
            boolean_values = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
            sample_str = [str(x).lower() for x in sample_values if x is not None]
            if sample_str and all(x in boolean_values for x in sample_str):
                transformations["column_type"] = "boolean"
                transformations["recommended_transformations"] = ["convert to boolean"]
                transformations["transformation_code"] = """# Convert to boolean
                                                        true_values = {'true', 't', 'yes', 'y', '1'}
                                                        df[column_name] = df[column_name].apply(
                                                            lambda x: True if isinstance(x, str) and x.lower() in true_values else 
                                                                    (False if isinstance(x, str) else bool(x))
                                                        )"""
                return transformations
        
        # Check for categorical data
        if unique_count / total_count < 0.2 and unique_count < 50:
            transformations["column_type"] = "categorical"
            transformations["recommended_transformations"] = ["convert to category"]
            transformations["transformation_code"] = "# Convert to category\ndf[column_name] = df[column_name].astype('category')"
            return transformations
        
        # Check for numeric strings
        if dtype == 'object' and len(sample_values) > 0:
            numeric_pattern = r'^[$€£¥]?\s*[\d,.]+\s*$'
            if all(re.match(numeric_pattern, str(x)) for x in sample_values if isinstance(x, str)):
                transformations["column_type"] = "numeric"
                transformations["recommended_transformations"] = ["convert to numeric"]
                transformations["transformation_code"] = """# Convert to numeric
                # Remove currency symbols and formatting
                df[column_name] = df[column_name].apply(
                    lambda x: re.sub(r'[^\\d.-]', '', str(x)) if isinstance(x, str) else x
                )
                df[column_name] = pd.to_numeric(df[column_name], errors='coerce')"""
                return transformations
        
        # Check for string cleaning
        if dtype == 'object' and len(sample_values) > 0:
            needs_cleaning = any(isinstance(x, str) and (x.strip() != x) for x in sample_values)
            if needs_cleaning:
                transformations["column_type"] = "string"
                transformations["recommended_transformations"] = ["trim whitespace"]
                transformations["transformation_code"] = """# Trim whitespace
                                                            df[column_name] = df[column_name].apply(
                                                                lambda x: x.strip() if isinstance(x, str) else x
                                                        )"""
                return transformations
        
        return transformations

# Full example with mock data
def demonstrate_preprocessing():
    """
    Demonstrate the AI data preprocessor with sample data.
    """
    # Sample data with various issues
    data = {
        "Customer ID": [1, 2, 3, 4, 5],
        "Name ": ["John Smith", "Jane Doe ", " Bob Johnson", "Alice Brown", "Charlie Davis"],
        "Date of Birth": ["1990-01-15", "1985-05-22", "1978-11-30", "1992-08-10", "1980-03-05"],
        "Registration Date": ["Jan 10, 2020", "Mar 15, 2020", "Dec 5, 2019", "Feb 28, 2020", "Apr 22, 2020"],
        "Active Status": ["Yes", "No", "Yes", "Yes", "No"],
        "Credit Score": ["750", "680", "820", "790", "700"],
        "Income": ["$50,000", "$65,000", "$45,000", "$72,000", "$58,000"],
        "Postal Code": ["90210", "10001", "60601", "75001", "94102"]
    }
    
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df.dtypes)
    print("\nSample data:")
    print(df.head())
    
    # Use dry run version for demonstration
    preprocessor = DryRunAIDataPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    print("\nProcessed DataFrame:")
    print(processed_df.dtypes)
    
    print("\nTransformations applied:")
    for col, transforms in preprocessor.get_summary().items():
        print(f"{col}: {transforms}")
    
    print("\nTransformation code examples:")
    for col, code in preprocessor.get_transformation_code().items():
        print(f"\n{col}:")
        print(code)

#demonstrate_preprocessing()
