# modules/data_preparation.py
import pandas as pd
import numpy as np

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
