import pandas as pd

def load_data(file):
    """
    Loads data from an uploaded file (CSV or Excel).
    """
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("Formato de arquivo não suportado. Use CSV ou Excel.")

def get_data_summary(df):
    """
    Returns a summary of the dataframe.
    """
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    return summary
