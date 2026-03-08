import os
import io
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

def generate_shap_explanation(model, X_train: pd.DataFrame, X_valid: pd.DataFrame = None, 
                              max_background_samples=100, task_type="Classification"):
    """
    Generates SHAP Global Feature Importance plot for Tabular data.
    """
    try:
        import shap
    except ImportError:
        warnings.warn("SHAP library not installed. Cannot generate explanations.")
        return None

    plt.switch_backend('Agg') # Ensure thread-safe rendering without GUI

    # 1. Determine background dataset (handle large data gracefully)
    bg_data = X_train
    if len(bg_data) > max_background_samples:
        bg_data = bg_data.sample(n=max_background_samples, random_state=42)
        
    evaluate_data = X_valid if X_valid is not None else bg_data
    if len(evaluate_data) > max_background_samples:
        evaluate_data = evaluate_data.sample(n=max_background_samples, random_state=42)

    # Convert non-numeric for generic shap handling if required by models
    # Depending on framework, categorical columns might need Ordinal/OneHot. 
    # For robust black-box generic explainer:
    
    explainer = None
    shap_values = None
    
    # 2. Heuristics to pick the right explainer
    model_type = str(type(model)).lower()
    
    try:
        if 'lgbm' in model_type or 'xgb' in model_type or 'catboost' in model_type or 'ensemble' in model_type:
            # TreeExplainer is fast for tree-based models and forests
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(evaluate_data)
            except Exception:
                pass # Fallback to generic

        if explainer is None:
            # For complex pipelines (like sklearn pipelines, PyCaret, generic wrappers)
            # Use KernelExplainer as a Black-Box proxy (requires a predict function)
            
            predict_fn = None
            if hasattr(model, "predict_proba") and "classification" in task_type.lower():
                predict_fn = lambda x: model.predict_proba(x)
            elif hasattr(model, "predict"):
                predict_fn = lambda x: model.predict(x)
            else:
                return None # Can't explain
                
            # KernelExplainer can be slow, hence the small bg_data
            explainer = shap.KernelExplainer(predict_fn, bg_data)
            shap_values = explainer.shap_values(evaluate_data)
            
    except Exception as e:
        warnings.warn(f"SHAP generation failed: {e}")
        return None

    # 3. Generate the Plot
    fig = plt.figure(figsize=(10, 6))
    
    try:
        # For multi-class, shap_values is a list. For regression/binary, it's an array.
        if isinstance(shap_values, list):
            # Take the shap values for the first class/positive class for overview
            shap.summary_plot(shap_values[1] if len(shap_values)>1 else shap_values[0], evaluate_data, show=False)
        else:
            shap.summary_plot(shap_values, evaluate_data, show=False)
            
        plt.tight_layout()
        return fig
    except Exception as e:
        warnings.warn(f"SHAP plot rendering failed: {e}")
        plt.close(fig)
        return None

def generate_cv_saliency_map(model, image_path: str, target_size=(224, 224), step=15, window_size=30):
    """
    Universal Occlusion Saliency Map for Black-Box CV Models (AutoGluon/AutoKeras).
    Instead of relying on internal hooks (which heavily abstracted AutoML layers hide),
    we slide a black box ('occlusion') across the image and measure the confidence drop.
    The regions that drop the confidence the most are the most salient (important) for the prediction.
    """
    try:
        from PIL import Image
        import cv2
    except ImportError:
        warnings.warn("Missing CV libraries (Pillow/OpenCV) for Saliency representation.")
        return None

    try:
        # 1. Load Original Image
        original_img = Image.open(image_path).convert('RGB')
        img_w, img_h = original_img.size
        
        # Determine the baseline prediction to see what class we are explaining
        # Since this is a generic AutoML predictor UI, we assume `model.predict_proba` gives a df or dict
        df_single = pd.DataFrame([{"image": image_path}])
        
        # Get base probabilities. 
        # Note: Depending on AutoGluon/AutoKeras formatting, the predict_proba method might vary.
        if hasattr(model, 'predict_proba'):
            base_probs = model.predict_proba(df_single)
            if isinstance(base_probs, pd.DataFrame):
                # Assuming top class 
                top_class = base_probs.iloc[0].idxmax()
                base_score = base_probs.iloc[0][top_class]
            else:
                top_class = np.argmax(base_probs[0])
                base_score = base_probs[0][top_class]
        else:
            warnings.warn("Model does not support predict_proba, Saliency Map cannot track confidence drops.")
            return None

        # 2. Build Saliency Map Array
        saliency_map = np.zeros((img_h, img_w))
        heatmap_counts = np.zeros((img_h, img_w))

        # We will create occluded images, save them temporarily, and batch-predict to find drops
        # For performance, we downsize the grid if the image is huge
        grid_step = step
        w_size = window_size
        
        # To avoid predicting 1000s of images, let's limit the grid
        if (img_h / step) * (img_w / step) > 200:
            grid_step = max(int(img_h/10), 10)
            w_size = int(grid_step * 1.5)

        occluded_paths = []
        coords = []
        
        tmp_dir = os.path.join("data_lake", "tmp_occlusion")
        os.makedirs(tmp_dir, exist_ok=True)
        img_arr_orig = np.array(original_img)

        # Generate Occluded Copies
        for y in range(0, img_h, grid_step):
            for x in range(0, img_w, grid_step):
                img_copy = img_arr_orig.copy()
                
                # Apply black box
                y1, y2 = max(0, y - w_size // 2), min(img_h, y + w_size // 2)
                x1, x2 = max(0, x - w_size // 2), min(img_w, x + w_size // 2)
                img_copy[y1:y2, x1:x2] = 0 # Occlude
                
                t_path = os.path.join(tmp_dir, f"occ_{y}_{x}.jpg")
                Image.fromarray(img_copy).save(t_path)
                occluded_paths.append(t_path)
                coords.append((y1, y2, x1, x2))

        # Predict all simultaneously
        df_batch = pd.DataFrame({"image": occluded_paths})
        
        try:
            batch_probs = model.predict_proba(df_batch)
        except Exception:
            warnings.warn("Batch probability prediction failed for occlusion map.")
            return None
            
        # Parse scores based on framework signature
        if isinstance(batch_probs, pd.DataFrame):
            scores = batch_probs[top_class].values
        else:
            scores = batch_probs[:, top_class] if len(batch_probs.shape) > 1 else batch_probs

        # 3. Calculate importance based on score drops
        for idx, (y1, y2, x1, x2) in enumerate(coords):
            drop = base_score - scores[idx]
            # If the score dropped, this region was important
            importance = max(0, drop)
            saliency_map[y1:y2, x1:x2] += importance
            heatmap_counts[y1:y2, x1:x2] += 1

        # Average overlaps
        heatmap_counts[heatmap_counts == 0] = 1
        saliency_avg = saliency_map / heatmap_counts

        # Normalize 0-255
        if np.max(saliency_avg) > 0:
            saliency_avg = (saliency_avg / np.max(saliency_avg)) * 255
        saliency_avg = np.uint8(saliency_avg)

        # 4. Generate visual overlay
        colormap = cv2.applyColorMap(saliency_avg, cv2.COLORMAP_JET)
        
        orig_cv = cv2.cvtColor(np.array(original_img), cv2.COLORRGB_BGR) # To match cv2
        final_overlay = cv2.addWeighted(orig_cv, 0.6, colormap, 0.4, 0)
        final_rgb = cv2.cvtColor(final_overlay, cv2.COLORBGR_RGB)
        
        # Cleanup
        for p in occluded_paths:
            try: os.remove(p)
            except: pass

        fig = plt.figure(figsize=(8, 8))
        plt.imshow(final_rgb)
        plt.title(f"XAI Occlusion Heatmap (Target: {top_class})")
        plt.axis('off')
        plt.tight_layout()
        
        return fig

    except Exception as e:
        warnings.warn(f"CV XAI generation failed: {e}")
        return None
