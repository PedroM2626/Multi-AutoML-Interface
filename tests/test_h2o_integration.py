#!/usr/bin/env python3
"""
Test script to verify H2O AutoML integration
"""

import pandas as pd
import numpy as np
from src.h2o_utils import train_h2o_model, load_h2o_model, initialize_h2o, cleanup_h2o
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'feature4': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature5': np.random.rand(n_samples) * 100
    }
    
    # Create target (classification)
    target = []
    for i in range(n_samples):
        # Simple rule to create target
        if data['feature1'][i] > 0.5 and data['feature2'][i] > 0:
            target.append('Class1')
        elif data['feature1'][i] < -0.5:
            target.append('Class2')
        else:
            target.append('Class0')
    
    data['target'] = target
    
    return pd.DataFrame(data)

def test_h2o_basic_functionality():
    """Test basic H2O functionality"""
    logger.info("Testing H2O basic functionality...")
    
    try:
        # Test H2O initialization
        h2o_instance = initialize_h2o()
        logger.info("✓ H2O initialization successful")
        
        # Test cleanup
        cleanup_h2o()
        logger.info("✓ H2O cleanup successful")
        
    except Exception as e:
        logger.error(f"✗ H2O basic functionality failed: {e}")
        return False
    
    return True

def test_h2o_training():
    """Test H2O AutoML training"""
    logger.info("Testing H2O AutoML training...")
    
    try:
        # Create sample data
        df = create_sample_data()
        logger.info(f"Created sample dataset with {len(df)} rows")
        
        # Test training with minimal parameters for quick test
        automl, run_id = train_h2o_model(
            train_data=df,
            target='target',
            run_name='test_h2o_run',
            max_runtime_secs=30,  # Quick test
            max_models=3,
            nfolds=2,
            balance_classes=True
        )
        
        logger.info(f"✓ H2O training successful. Run ID: {run_id}")
        
        # Test model loading
        loaded_model = load_h2o_model(run_id)
        logger.info("✓ H2O model loading successful")
        
        # Test prediction
        test_data = df.head(10).drop(columns=['target'])
        from src.h2o_utils import predict_with_h2o
        predictions = predict_with_h2o(loaded_model, test_data)
        logger.info(f"✓ H2O prediction successful. Predictions shape: {predictions.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ H2O training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mlflow_integration():
    """Test MLflow integration"""
    logger.info("Testing MLflow integration...")
    
    try:
        import mlflow
        
        # Check if experiment exists
        experiment = mlflow.get_experiment_by_name("H2O_Experiments")
        if experiment:
            logger.info("✓ H2O_Experiments found in MLflow")
            
            # List runs
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            logger.info(f"✓ Found {len(runs)} runs in H2O_Experiments")
            
            if not runs.empty:
                logger.info("Latest runs:")
                print(runs[['run_id', 'status', 'start_time']].tail())
        else:
            logger.warning("H2O_Experiments not found in MLflow")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ MLflow integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting H2O AutoML integration tests...")
    
    tests = [
        ("Basic H2O Functionality", test_h2o_basic_functionality),
        ("H2O Training", test_h2o_training),
        ("MLflow Integration", test_mlflow_integration),
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results[test_name] = result
            status = "✓ PASSED" if result else "✗ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ✗ FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! H2O AutoML integration is working correctly.")
    else:
        logger.warning("⚠️  Some tests failed. Please check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
