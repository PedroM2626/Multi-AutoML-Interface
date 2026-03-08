import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image
import time
import queue

from src.autogluon_utils import train_model as ag_train
from src.autokeras_utils import run_autokeras_experiment
from src.modelsearch_utils import run_modelsearch_experiment

def setup_dummy_images(base_dir):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(os.path.join(base_dir, 'class_a'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'class_b'), exist_ok=True)
    
    # Create valid dummy images
    for i in range(5):
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(os.path.join(base_dir, 'class_a', f'img_{i}.jpg'))
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(os.path.join(base_dir, 'class_b', f'img_{i}.jpg'))
        
    print(f"[{base_dir}] Dummy images created.")
    return pd.DataFrame([{"Image_Directory": base_dir}])

def main():
    base_dir = "test_cv_dummy_data"
    df = setup_dummy_images(base_dir)
    
    log_q = queue.Queue()
    tasks = [
        "Computer Vision - Image Classification",
        "Computer Vision - Object Detection",
        "Computer Vision - Image Segmentation",
        "Multi-Label Classification"
    ]
    
    print("=== Testing AutoGluon ===")
    for task in tasks:
        print(f"\\n--- AutoGluon Task: {task} ---")
        try:
            ag_train(
                train_data=df, 
                target="label", 
                run_name=f"ag_test_{int(time.time())}", 
                time_limit=10, 
                task_type=task
            )
            print(f"\\nSuccess for {task}")
        except Exception as e:
            print(f"\\nFailed for {task}: {e}")
            
    print("\\n=== Testing AutoKeras ===")
    for task in tasks:
        if task == "Multi-Label Classification": continue
        print(f"\\n--- AutoKeras Task: {task} ---")
        try:
            run_autokeras_experiment(
                train_data=df, 
                target="label", 
                run_name=f"ak_test_{int(time.time())}", 
                time_limit=10, 
                task_type=task,
                log_queue=log_q
            )
            print(f"\\nSuccess for {task}")
        except Exception as e:
            print(f"\\nFailed for {task}: {e}")
            
    print("\\n=== Testing Model Search ===")
    for task in tasks:
        if task == "Multi-Label Classification": continue
        print(f"\\n--- Model Search Task: {task} ---")
        try:
            run_modelsearch_experiment(
                train_data=df, 
                target="label", 
                run_name=f"ms_test_{int(time.time())}", 
                task_type=task,
                log_queue=log_q
            )
            print(f"\\nSuccess for {task}")
        except Exception as e:
            print(f"\\nFailed for {task}: {e}")

if __name__ == "__main__":
    main()
