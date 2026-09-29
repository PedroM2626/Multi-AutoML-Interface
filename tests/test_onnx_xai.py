"""ONNX export and the tabular XAI path.

Two of these needed an interpreter with the converters installed, so they skip in the lean PR
gate and run in the nightly, which installs requirements.txt - the same file the installers
build their runtime from.
"""
import ast
import importlib.util
import warnings

import numpy as np
import pandas as pd
import pytest

needs_onnx = pytest.mark.skipif(
    importlib.util.find_spec("skl2onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="skl2onnx/onnxruntime not installed",
)


def test_tabular_xai_does_not_need_opencv():
    """The module imported cv2 at the top, so the SHAP explanation died wherever OpenCV was
    absent - the computer-vision path imports it locally and does not need this."""
    tree = ast.parse(open("src/xai_utils.py", encoding="utf-8").read())
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.add(node.module.split(".")[0])
    assert "cv2" not in top_level


@needs_onnx
def test_a_scikit_learn_model_round_trips_through_onnx(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    from src.onnx_utils import export_to_onnx, load_onnx_session, predict_onnx

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)

    path = export_to_onnx(model, "flaml", "target", str(tmp_path / "rf.onnx"), input_sample=X[:1])

    predictions = predict_onnx(load_onnx_session(path), X)
    assert len(predictions) == len(X)
    assert (np.asarray(predictions).astype(int) == y.to_numpy()).mean() > 0.8


@needs_onnx
@pytest.mark.skipif(importlib.util.find_spec("lightgbm") is None, reason="lightgbm not installed")
def test_a_boosted_tree_learner_reports_the_missing_converter(tmp_path):
    """skl2onnx has no LightGBM converter; FLAML's default learner used to fail as a warning
    logged inside a training thread, which the user never saw."""
    import lightgbm as lgb

    from src.onnx_utils import export_to_onnx

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int)
    model = lgb.LGBMClassifier(n_estimators=4, verbose=-1).fit(X, y)

    with pytest.raises(NotImplementedError, match="LGBMClassifier"):
        export_to_onnx(model, "flaml", "target", str(tmp_path / "lgbm.onnx"), input_sample=X[:1])


@needs_onnx
def test_the_flaml_automl_wrapper_is_unwrapped_before_export(tmp_path):
    """The Experiments button hands over the AutoML object; skl2onnx needs the inner estimator."""
    if importlib.util.find_spec("flaml") is None:
        pytest.skip("flaml not installed")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytest.importorskip("lightgbm", reason="FLAML AutoML needs lightgbm to train")
        from flaml import AutoML

        from src.onnx_utils import export_to_onnx

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 4)), columns=[f"f{i}" for i in range(4)])
    y = (X["f0"] > 0).astype(int)
    automl = AutoML()
    automl.fit(X_train=X, y_train=y, task="classification", time_budget=4,
               estimator_list=["rf"], seed=0, verbose=0, log_file_name="")

    path = export_to_onnx(automl, "flaml", "target", str(tmp_path / "automl.onnx"), input_sample=X[:1])

    assert path.endswith("automl.onnx")
