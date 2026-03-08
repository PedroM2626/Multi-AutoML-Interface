import os
import shutil
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import time
import queue

# Emulate AutoGluon dataset helpers if possible
try:
    from autogluon.core.utils.loaders import load_zip
    AG_AVAILABLE = True
except ImportError:
    AG_AVAILABLE = False

from src.autogluon_utils import train_model as ag_train
from src.autokeras_utils import run_autokeras_experiment
from src.modelsearch_utils import run_modelsearch_experiment

def setup_dummy_classification(base_dir="test_cv_cls"):
    """Create dummy images for standard Image Classification"""
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(os.path.join(base_dir, 'cat'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dog'), exist_ok=True)
    
    for i in range(3):
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(os.path.join(base_dir, 'cat', f'img_{i}.jpg'))
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(os.path.join(base_dir, 'dog', f'img_{i}.jpg'))
    return pd.DataFrame([{"Image_Directory": base_dir}])

def setup_dummy_multilabel(base_dir="test_cv_multilabel"):
    """Create dummy images and a properly formatted dataframe for Multi-Label"""
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    data = []
    
    for i in range(20):
        path = os.path.join(base_dir, f'img_{i}.jpg')
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        # Random multi-label combinations
        data.append({"image": path, "cat": 1 if i%2==0 else 0, "dog": 1 if i%3==0 else 0, "bird": 1 if i%5==0 else 0})
        
    df = pd.DataFrame(data)
    # We will tell it the target is 'cat' just for checking but actually in multi-label we might pass a list of targets
    return df

def setup_dummy_segmentation(base_dir="test_cv_seg"):
    """Create dummy images and masks for Semantic Segmentation"""
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    img_dir = os.path.join(base_dir, 'images')
    mask_dir = os.path.join(base_dir, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    data = []
    for i in range(15):
        img_path = os.path.join(img_dir, f'img_{i}.jpg')
        mask_path = os.path.join(mask_dir, f'mask_{i}.png')
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(img_path)
        Image.fromarray(np.random.randint(0, 3, (32, 32), dtype=np.uint8)).save(mask_path)
        data.append({"image": img_path, "label": mask_path})
        
    return pd.DataFrame(data)

def setup_dummy_detection(base_dir="test_cv_det"):
    """Create dummy images and COCO-like format for Object Detection"""
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    data = []
    for i in range(15):
        img_path = os.path.join(base_dir, f'img_{i}.jpg')
        Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)).save(img_path)
        data.append({"image": img_path, "rois": [{"class": "cat", "bbox": [5, 5, 20, 20]}]})
        
    return pd.DataFrame(data)


def main():
    print("Preparing Datasets...")
    df_cls = setup_dummy_classification()
    df_ml = setup_dummy_multilabel()
    df_seg = setup_dummy_segmentation()
    df_det = setup_dummy_detection()
    
    log_q = queue.Queue()
    
    print("\\n" + "="*50)
    print("1. AUTOGLUON TESTS")
    print("="*50)
    
    # AutoGluon Multi-Label Classification
    print("\\n-- AUTOGLUON: Computer Vision - Multi-Label Classification --")
    try:
        # For AutoGluon MultiModal, to do multilabel we can setup problem_type="multiclass" natively via Tabular/MultiModal or explicitly.
        # But we'll use task_type="Computer Vision - Image Classification" and just pass subset of targets or let it throw unsupported if it wants target=1col
        # Let's pass target="cat" just to see it complete over one column since we only pass one in the UI usually
        ag_train(train_data=df_ml, target="cat", run_name="ag_ml_test", time_limit=10, task_type="Computer Vision - Image Classification")
        print(">> SUCCESS: AutoGluon passed multi-label/binary test.")
    except Exception as e:
        print(f">> FAILED: AutoGluon Multi-Label. Error: {e}")

    # AutoGluon Object Detection
    print("\\n-- AUTOGLUON: Computer Vision - Object Detection --")
    try:
        ag_train(train_data=df_det, target="rois", run_name="ag_det_test", time_limit=10, task_type="Computer Vision - Object Detection")
        print(">> SUCCESS: AutoGluon passed object detection test.")
    except Exception as e:
        print(f">> FAILED: AutoGluon Object Detection. Error: {e}")

    # AutoGluon Image Segmentation
    print("\\n-- AUTOGLUON: Computer Vision - Image Segmentation --")
    try:
        ag_train(train_data=df_seg, target="label", run_name="ag_seg_test", time_limit=10, task_type="Computer Vision - Image Segmentation")
        print(">> SUCCESS: AutoGluon passed image segmentation test.")
    except Exception as e:
        print(f">> FAILED: AutoGluon Semantic Segmentation. Error: {e}")


    print("\\n" + "="*50)
    print("2. AUTOKERAS TESTS")
    print("="*50)
    
    print("\\n-- AUTOKERAS: Computer Vision - Image Classification --")
    try:
        run_autokeras_experiment(train_data=df_cls, target="label", run_name="ak_cls_test", time_limit=10, task_type="Computer Vision - Image Classification", log_queue=log_q)
        print(">> SUCCESS: AutoKeras Image Classification.")
    except Exception as e:
        print(f">> FAILED: AutoKeras Image Classification. Error: {e}")
        
    print("\\n-- AUTOKERAS: Unsupported Tasks (Should catch NotImplementedError) --")
    for task in ["Computer Vision - Object Detection", "Computer Vision - Image Segmentation"]:
        try:
            run_autokeras_experiment(train_data=df_cls, target="label", run_name="ak_err_test", time_limit=10, task_type=task, log_queue=log_q)
        except NotImplementedError as e:
            print(f">> SUCCESS: Caught expected NotImplementedError for AutoKeras {task}.")
        except Exception as e:
            print(f">> FAILED: Unexpected error for {task}: {e}")

            
    print("\\n" + "="*50)
    print("3. MODEL SEARCH TESTS")
    print("="*50)
    
    print("\\n-- MODEL SEARCH: Computer Vision - Image Classification --")
    try:
        run_modelsearch_experiment(train_data=df_cls, target="label", run_name="ms_cls_test", task_type="Computer Vision - Image Classification", log_queue=log_q)
        print(">> SUCCESS: Model Search Image Classification.")
    except Exception as e:
        print(f">> FAILED: Model Search. Error: {e}")

if __name__ == "__main__":
    main()
