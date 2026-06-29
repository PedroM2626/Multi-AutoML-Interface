import numpy as np
import pandas as pd
from src.processor import AutoMLDataProcessor
from src.flaml_utils import train_flaml_model
from src.autogluon_utils import train_model as train_autogluon

def test_processor():
    print("Testing AutoMLDataProcessor...")
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10, freq='D'),
        'text': ["Hello world", "This is text", "Another sentence", "Machine learning", "AutoML ops", 
                 "Hello world", "This is text", "Another sentence", "Machine learning", "AutoML ops"],
        'feat1': [1.0, 2.0, 1.5, 3.0, 2.5, 1.0, 2.0, 1.5, 3.0, 2.5],
        'target1': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'target2': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    })
    
    # 1. Temporal & NLP
    processor = AutoMLDataProcessor(
        target_column='target1',
        task_type='classification',
        date_col='date',
        is_time_series=True
    )
    X, y = processor.fit_transform(df, nlp_cols=['text'])
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    assert X.shape[0] > 0
    assert y is not None

    # 2. Multi-Task
    processor_multi = AutoMLDataProcessor(
        target_column=['target1', 'target2'],
        task_type='classification'
    )
    X_m, y_m = processor_multi.fit_transform(df)
    print(f"Multi-target shape: {y_m.shape}")
    assert y_m.shape[1] == 2

    # 3. Semi-supervised
    df_semi = df.copy()
    df_semi.loc[5:, 'target1'] = -1  # unlabeled
    processor_semi = AutoMLDataProcessor(
        target_column='target1',
        task_type='classification',
        semi_supervised=True
    )
    X_s, y_s = processor_semi.fit_transform(df_semi)
    assert -1 in y_s.values
    print("Processor tests passed!")

def test_flaml_single_target():
    print("Testing FLAML Single Target...")
    df = pd.DataFrame({
        'feat1': np.random.randn(20),
        'feat2': np.random.randn(20),
        'target1': np.random.randint(0, 2, 20)
    })
    # Run
    automl, run_id = train_flaml_model(
        train_data=df,
        target='target1',
        run_name='test_flaml_single',
        time_budget=5,
        task='classification'
    )
    assert automl is not None
    assert run_id is not None
    preds = automl.predict(df.drop(columns=['target1']))
    assert preds.shape == (20,)
    print("FLAML Single Target test passed!")

def test_autogluon_single_target():
    print("Testing AutoGluon Single Target...")
    df = pd.DataFrame({
        'feat1': np.random.randn(20),
        'feat2': np.random.randn(20),
        'target1': np.random.randint(0, 2, 20)
    })
    # Run autogluon
    train_autogluon(
        train_data=df,
        target='target1',
        run_name='test_ag_single',
        time_limit=5,
        task_type='Classification',
        data_category='Tabular'
    )
    print("AutoGluon Single Target test passed!")

if __name__ == "__main__":
    test_processor()
    test_flaml_single_target()
    test_autogluon_single_target()
    print("All Multi-AutoML-Interface tests passed!")
