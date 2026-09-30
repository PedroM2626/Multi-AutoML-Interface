"""End-to-end proof that every row the catalog offers trains and scores in this interpreter.

Each case calls the same engine function the UI dispatches to, on a dataset small enough to
train in seconds, and skips when the engine (or its extra) is not installed here. Marked
``engine`` because it imports real AutoML libraries: the quick PR gate never runs it, the
nightly job installs the full engine lock and runs it.
"""
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.data_utils import cv_label_columns
from src.task_catalog import TASK_FRAMEWORK_MAP, framework_available, preload_torch_before_sklearn

# app.py runs this before it imports anything that reaches scikit-learn, and the order matters
# here for the same reason: on the PyCaret-era lock, scikit-learn first makes torch's c10.dll
# die with WinError 1114 and autogluon.multimodal never imports.
preload_torch_before_sklearn()

from src.orchestrator import UniversalAutoMLOrchestrator  # noqa: E402
from src.prediction_service import run_predictions  # noqa: E402
from src import autogluon_utils  # noqa: E402

pytestmark = pytest.mark.engine

TIME_LIMIT = 20
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def workdir():
    path = tempfile.mkdtemp(prefix="engine-matrix-")
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _classification_frame(rows: int = 120) -> pd.DataFrame:
    scores = RNG.normal(size=(rows, 4))
    logits = scores @ np.array([1.6, -1.1, 0.7, 0.3])
    labels = np.where(logits > 0, "cat", "dog")
    frame = pd.DataFrame(scores, columns=[f"x{i}" for i in range(4)])
    frame["y"] = labels
    return frame


def _regression_frame(rows: int = 120) -> pd.DataFrame:
    scores = RNG.normal(size=(rows, 3))
    frame = pd.DataFrame(scores, columns=[f"x{i}" for i in range(3)])
    frame["y"] = scores @ np.array([2.0, -1.0, 0.5]) + RNG.normal(scale=0.2, size=rows)
    return frame


def _multi_target_frame(rows: int = 120) -> pd.DataFrame:
    scores = RNG.normal(size=(rows, 4))
    frame = pd.DataFrame(scores, columns=[f"x{i}" for i in range(4)])
    frame["a"] = (scores[:, 0] + RNG.normal(scale=0.3, size=rows) > 0).astype(int)
    frame["b"] = (scores[:, 1] - scores[:, 2] + RNG.normal(scale=0.3, size=rows) > 0).astype(int)
    return frame


def _features_only_frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame(RNG.normal(size=(rows, 4)), columns=[f"x{i}" for i in range(4)])


def _lagged_forecast_frame(rows: int = 150) -> pd.DataFrame:
    """What the Tabular Forecast row hands the engine: the processor has already built the
    lag features, so the run is a supervised regression over shifted history."""
    base = np.cumsum(RNG.normal(size=rows)) + 20.0
    frame = pd.DataFrame({
        "y_lag1": np.r_[np.nan, base[:-1]],
        "y_lag2": np.r_[np.nan, np.nan, base[:-2]],
        "rolling_mean": pd.Series(base).rolling(3, min_periods=1).mean().to_numpy(),
    })
    frame["y"] = np.r_[np.nan, base[:-1]]
    return frame.dropna().reset_index(drop=True)


def _sequential_forecast_frame(rows: int = 150) -> pd.DataFrame:
    """Raw time-ordered table for the Sequential row: date column plus the value to forecast."""
    dates = pd.date_range("2020-01-01", periods=rows, freq="D")
    seasonal = np.sin(np.arange(rows) / 7.0 * 2 * np.pi) * 3.0
    frame = pd.DataFrame({"date": dates, "value": 30.0 + seasonal + RNG.normal(scale=0.4, size=rows)})
    frame["exog"] = RNG.normal(size=rows)
    return frame


def _ranking_frame(rows: int = 150) -> pd.DataFrame:
    """Groups of items with integer relevance grades, which is what FLAML's rank task wants."""
    groups = np.repeat(np.arange(rows // 5), 5)
    scores = RNG.normal(size=rows) + groups * 0.1
    frame = pd.DataFrame({"query": groups, "x0": scores, "x1": RNG.normal(size=rows)})
    frame["y"] = np.clip((scores * 2).round().astype(int), 0, 4)
    return frame


def _text_frame(rows: int = 120) -> pd.DataFrame:
    """Sentences that vary per row. AutoGluon types a text column from its cardinality and word
    count, so a column of six repeated sentences becomes categorical, and MultiModalPredictor
    then refuses with "No model is available for this dataset."
    """
    openers = ["the film", "this movie", "the story", "the cast", "the score"]
    positives = ["was wonderful", "is a delight", "shines with real warmth", "is beautifully shot"]
    negatives = ["was a dull mess", "never found its pace", "wasted its good cast", "felt lifeless"]
    closers = ["from the first scene", "with a sharp script", "and I would watch it again", "despite the length"]
    texts, labels = [], []
    for index in range(rows):
        positive = index % 2 == 0
        verdict = positives[index % 4] if positive else negatives[(index + 1) % 4]
        texts.append(f"{openers[index % 5]} {verdict} {closers[(index * 3) % 4]} ({index})")
        labels.append("positive" if positive else "negative")
    return pd.DataFrame({"review": texts, "sentiment": labels})


def _text_regression_frame(rows: int = 120) -> pd.DataFrame:
    texts = [f"item {i} description with words that carry a rating signal" for i in range(rows)]
    return pd.DataFrame({"review": texts, "rating": RNG.integers(1, 6, size=rows).astype(float)})


def _image_directory(workdir: str, classes: int = 3, per_class: int = 5) -> str:
    """Folder-labelled images: the shape the CV upload produces and build_image_df walks."""
    from PIL import Image

    root = os.path.join(workdir, f"images_{classes}_{per_class}")
    if not os.path.isdir(root):
        palette = RNG.integers(0, 255, size=(classes, 3))
        for index in range(classes):
            folder = os.path.join(root, f"class_{index}")
            os.makedirs(folder, exist_ok=True)
            colour = tuple(int(value) for value in palette[index])
            for shot in range(per_class):
                noise = RNG.integers(0, 40, size=(32, 32, 3)).astype("uint8")
                tinted = np.clip(np.array(Image.new("RGB", (32, 32), colour)) + noise, 0, 255)
                Image.fromarray(tinted.astype("uint8")).save(os.path.join(folder, f"img_{shot}.png"))
    return root


def _image_frame(workdir: str) -> pd.DataFrame:
    return pd.DataFrame({"Image_Directory": [_image_directory(workdir)]})


def _annotated_image_frame(workdir: str) -> pd.DataFrame:
    """Exactly what `load_data` returns for an annotated CV dataset: image names relative to the
    folder, one 0/1 column per label, and the folder itself in Image_Directory - which is how
    train_model knows to resolve the paths."""
    images = _image_directory(workdir)
    rows = []
    for class_index, folder in enumerate(sorted(os.listdir(images))):
        for shot, name in enumerate(sorted(os.listdir(os.path.join(images, folder)))):
            rows.append({
                "image": os.path.join(folder, name),
                "warm": class_index % 2,
                "bright": shot % 2,
            })
    frame = pd.DataFrame(rows)
    frame["Image_Directory"] = images
    return frame


def _multimodal_frame(workdir: str, task_type: str = "Classification") -> pd.DataFrame:
    images = _image_directory(workdir)
    files = [
        os.path.join(folder, name)
        for folder, _, names in os.walk(images) for name in names
    ]
    rows = min(len(files), 30)
    frame = pd.DataFrame({
        "image": files[:rows],
        "note": [f"swatch number {index} with a few words about its colour" for index in range(rows)],
        "x0": RNG.normal(size=rows),
    })
    if task_type == "Regression":
        frame["y"] = RNG.normal(size=rows) * 10.0
    else:
        frame["y"] = [f"class_{index % 3}" for index in range(rows)]
    return frame


# (data_category, task_type, framework) -> kwargs, in the shape app.py builds them.
def build_kwargs(framework: str, data_category: str, task_type: str, workdir: str, run_name: str):
    common = {"run_name": run_name}
    if framework == "AutoGluon":
        if data_category == "Computer Vision":
            if task_type == "Multi-Label Classification":
                frame = _annotated_image_frame(workdir)
                target = cv_label_columns(frame.columns)
            else:
                frame = _image_frame(workdir)
                target = "label"
        elif data_category == "Multimodal":
            frame = _multimodal_frame(workdir, task_type)
            target = "y"
        elif task_type == "Multi-Task Classification":
            frame = _multi_target_frame()
            target = ["a", "b"]
        elif task_type == "Multi-Label Classification":
            frame = _multi_target_frame()
            target = ["a", "b"]
        elif task_type == "Forecast":
            frame = _lagged_forecast_frame()
            target = "y"
        elif data_category == "Text" or task_type == "Regression":
            frame = _text_regression_frame() if task_type == "Regression" else _text_frame()
            target = "rating" if task_type == "Regression" else "sentiment"
        elif task_type == "Classification":
            frame = _classification_frame()
            target = "y"
            frame = frame.astype({"y": "category"})
        else:
            frame = _regression_frame()
            target = "y"
        return dict(
            train_data=frame, target=target, valid_data=None, test_data=None,
            time_limit=TIME_LIMIT, presets="medium_quality", seed=42, cv_folds=0,
            task_type=f"{data_category} - {task_type}" if data_category == "Computer Vision" else task_type,
            data_category=data_category,
            multimodal_text_columns=["review"] if data_category == "Text" else (["note"] if data_category == "Multimodal" else []),
            multimodal_image_columns=["image"] if data_category == "Multimodal" else [],
            **common,
        )

    if framework == "FLAML":
        # Metric names are the UI's: flaml's registry keys are lower case, and the rank and
        # ts_forecast selectors only offer "auto".
        if data_category == "Sequential":
            frame, task, metric = _sequential_forecast_frame(), "ts_forecast", "auto"
            extra = {"time_col": "date", "period": 1}
        elif task_type == "Ranking":
            frame, task, metric = _ranking_frame(), "rank", "auto"
            extra = {"group_col": "query"}
        elif task_type == "Forecast":
            frame, task, metric = _lagged_forecast_frame(), "regression", "rmse"
            extra = {}
        elif task_type in ("Multi-Task Classification", "Multi-Label Classification"):
            frame, task, metric = _multi_target_frame(), "classification", "accuracy"
            extra = {}
        elif task_type == "Regression":
            frame, task, metric = _regression_frame(), "regression", "rmse"
            extra = {}
        else:
            frame, task, metric = _classification_frame(), "classification", "accuracy"
            extra = {}
        target = "y" if "y" in frame.columns else frame.columns[-1]
        return dict(
            train_data=frame, target=target, valid_data=None, test_data=None,
            time_budget=8, task=task, metric=metric, estimator_list=["lgbm"], seed=42,
            cv_folds=2, n_jobs=1, time_col=extra.get("time_col"), period=extra.get("period"),
            group_col=extra.get("group_col"), **common,
        )

    if framework == "H2O AutoML":
        if task_type in ("Multi-Task Classification", "Multi-Label Classification"):
            frame = _multi_target_frame()
            target = "a"
        elif task_type == "Regression":
            frame, target = _regression_frame(), "y"
        else:
            frame, target = _classification_frame(), "y"
        return dict(
            train_data=frame, target=target, valid_data=None, test_data=None,
            max_runtime_secs=15, max_models=3, nfolds=2, balance_classes=False,
            seed=42, sort_metric=None, exclude_algos=[], **common,
        )

    if framework == "PyCaret":
        if task_type == "Anomaly Detection":
            frame, target = _features_only_frame(), None
        elif task_type == "Clustering":
            frame, target = _features_only_frame(), None
        elif task_type in ("Forecast", "Time Series Forecasting"):
            frame, target = _sequential_forecast_frame(), "value"
        elif task_type == "Regression":
            frame, target = _regression_frame(), "y"
        elif task_type in ("Multi-Task Classification", "Multi-Label Classification"):
            frame, target = _multi_target_frame(), "a"
        else:
            frame, target = _classification_frame(), "y"
        return dict(
            train_df=frame, target_col=target, val_df=None, time_limit=TIME_LIMIT,
            task_type=task_type, fh=2 if task_type == "Forecast" else None,
            seasonal_period=7 if task_type == "Forecast" else None,
            time_col="date" if frame is not None and "date" in getattr(frame, "columns", []) else None,
            n_jobs=1, log_queue=None, **common,
        )

    if framework == "Lale":
        if task_type in ("Multi-Task Classification", "Multi-Label Classification"):
            frame, target = _multi_target_frame(), "a"
        elif task_type == "Regression":
            frame, target = _regression_frame(), "y"
        else:
            frame, target = _classification_frame(), "y"
        return dict(
            train_df=frame, target_col=target, val_df=None, time_limit=TIME_LIMIT,
            task_type=task_type, log_queue=None, **common,
        )

    if framework == "TPOT":
        if task_type == "Regression":
            frame, target = _regression_frame(), "y"
        elif task_type in ("Multi-Task Classification", "Multi-Label Classification"):
            frame, target = _multi_target_frame(), "a"
        else:
            frame, target = _classification_frame(), "y"
        return dict(
            df=frame, target_column=target, valid_data=None, test_data=None,
            generations=2, population_size=4, cv=2, scoring=None, max_time_mins=1,
            max_eval_time_mins=1, random_state=42, verbosity=0, n_jobs=1,
            config_dict=None, tfidf_max_features=None, tfidf_ngram_range=None, **common,
        )

    raise AssertionError(f"no fixture knows how to drive {framework}")


def _catalog_rows():
    for (data_category, task_type), frameworks in TASK_FRAMEWORK_MAP.items():
        for framework in frameworks:
            yield pytest.param(framework, data_category, task_type,
                               id=f"{data_category}/{task_type}/{framework}")


def _prediction_input(framework: str, data_category: str, kwargs: dict, workdir: str):
    """What the Prediction page can hand the trained model: a small table of the same features.

    A computer-vision run trains from a folder or an annotations table, but the engine predicts
    from image paths, which is what a prediction CSV carries - so the CV rows are scored that way.
    """
    if data_category == "Computer Vision":
        root = _image_directory(workdir)
        files = [
            os.path.join(folder, name)
            for folder, _, names in os.walk(root) for name in names
        ][:3]
        return pd.DataFrame({"image": files})

    frame = None
    for key in ("train_data", "train_df", "df"):
        frame = kwargs.get(key)
        if frame is not None:
            break
    return frame.head(4).copy()


@pytest.mark.parametrize("framework,data_category,task_type", list(_catalog_rows()))
def test_catalog_row_trains_and_predicts(framework, data_category, task_type, workdir):
    if not framework_available(framework, data_category):
        pytest.skip(f"{framework} is not installed in this interpreter")

    if (framework == "AutoGluon" and data_category == "Tabular"
            and autogluon_utils._pandas_lacks_the_downcasting_option()):
        # The all-engine lock cannot satisfy both: PyCaret 3.3.2 pins pandas<2.2 and AutoGluon's
        # tabular learner opens an option that only exists from 2.2. train_model refuses the run
        # with that message; the matrix records it instead of pretending the row is untested.
        pytest.xfail("AutoGluon's tabular fit needs pandas >= 2.2, which PyCaret's pin forbids")

    run_name = f"matrix_{framework.lower()}_{data_category}_{task_type}".replace(" ", "_")[:60]
    kwargs = build_kwargs(framework, data_category, task_type, workdir, run_name)
    orchestrator = UniversalAutoMLOrchestrator(framework, dict(kwargs, dataset_path=None))

    result = orchestrator.run_synchronously()
    assert result is not None, f"{framework} returned nothing for {data_category}/{task_type}"

    predictor = result["predictor"] if isinstance(result, dict) else result[0]
    model_type = UniversalAutoMLOrchestrator.FRAMEWORK_MAPPINGS[framework][0]
    target = kwargs.get("target") or kwargs.get("target_col") or kwargs.get("target_column")

    predict_df = _prediction_input(framework, data_category, kwargs, workdir)
    result_df, predict_input = run_predictions(
        predictor=predictor,
        model_type=model_type,
        predict_df=predict_df,
        target_col=target,
        training_df=None,
        task_type=task_type,
    )

    assert len(result_df) == len(predict_df)
    assert len(predict_input) == len(predict_df)
    assert any(
        column == "Predictions" or column.startswith("Prediction_")
        for column in result_df.columns
    ), f"{framework} produced no prediction column for {data_category}/{task_type}: {list(result_df.columns)}"
