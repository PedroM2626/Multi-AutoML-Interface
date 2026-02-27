import os
import subprocess
import hashlib
import time
import pandas as pd

def load_data(file):
    """
    Loads data from an uploaded file (CSV or Excel) or a disk path.
    """
    is_path = isinstance(file, str)
    filename = file if is_path else file.name
    
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
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

def init_dvc():
    """
    Initializes a DVC repository in the current directory if it doesn't exist.
    """
    if not os.path.exists(".dvc"):
        try:
            subprocess.run(["dvc", "init"], check=True, capture_output=True)
            print("DVC repository initialized successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to initialize DVC: {e}")
        except FileNotFoundError:
            print("DVC is not installed or not in PATH.")

def save_to_data_lake(df, filename_prefix="dataset"):
    """
    Saves a DataFrame to the local data lake, tracks it with DVC, and returns its metadata hash.
    """
    data_lake_dir = os.path.join("data_lake", "raw")
    os.makedirs(data_lake_dir, exist_ok=True)
    
    # Generate unique filename based on time
    timestamp = int(time.time())
    file_path = os.path.join(data_lake_dir, f"{filename_prefix}_{timestamp}.csv")
    
    # Save the dataframe
    df.to_csv(file_path, index=False)
    
    # Add to DVC
    dvc_hash = "unknown_hash"
    try:
        init_dvc() # Ensure DVC is initialized
        subprocess.run(["dvc", "add", file_path], check=True, capture_output=True)
        # Assuming dvc add creates a .dvc file, we can potentially read it or just use the filename hash as a proxy
        dvc_file_path = file_path + ".dvc"
        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                content = f.read()
                # Simple extraction of md5 from the dvc file if available
                import re
                match = re.search(r'md5:\s*([a-fA-F0-9]+)', content)
                if match:
                    dvc_hash = match.group(1)
    except Exception as e:
        print(f"DVC error: {e}")
        # Fallback to computing standard MD5 if DVC fails
        with open(file_path, "rb") as f:
            dvc_hash = hashlib.md5(f.read()).hexdigest()
            
    return file_path, dvc_hash, dvc_hash[:8]

def get_data_lake_files():
    """
    Retrieves all available datasets in the data lake.
    """
    data_lake_dir = os.path.join("data_lake", "raw")
    if not os.path.exists(data_lake_dir):
        return []
    
    files = []
    for f in os.listdir(data_lake_dir):
        if f.endswith(('.csv', '.xls', '.xlsx')):
            files.append(os.path.join(data_lake_dir, f))
    
    # Sort files by creation time descending (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files

def get_dvc_hash(file_path):
    """
    Extracts the DVC hash corresponding to a specific file.
    """
    dvc_hash = "unknown_hash"
    dvc_file_path = file_path + ".dvc"
    if os.path.exists(dvc_file_path):
        with open(dvc_file_path, "r") as f:
            content = f.read()
            import re
            match = re.search(r'md5:\s*([a-fA-F0-9]+)', content)
            if match:
                dvc_hash = match.group(1)
                return dvc_hash, dvc_hash[:8]
                
    # Fallback to computing MD5
    try:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                dvc_hash = hashlib.md5(f.read()).hexdigest()
    except:
        pass
        
    return dvc_hash, dvc_hash[:8]
