#!/usr/bin/env python3
"""
Teste para verificar o tratamento de NaN no TPOT
"""

import pandas as pd
import numpy as np
import sys
import os
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_tpot_with_nan_data():
    """Testar TPOT com dados contendo NaN"""
    logger.info("🧪 Testando TPOT com dados contendo NaN...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados com NaN
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': [np.nan if i % 10 == 0 else x for i, x in enumerate(np.random.randn(n_samples))],
            'feature3': ['A' if i % 3 == 0 else 'B' if i % 3 == 1 else np.nan for i in range(n_samples)],
            'feature4': ['text data' if i % 5 == 0 else np.nan for i in range(n_samples)],
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dados criados com NaN: {df.isnull().sum().sum()} valores nulos")
        
        target_column = 'target'
        run_name = f"tpot_test_nan_{int(time.time())}"
        
        # Parâmetros para teste rápido
        params = {
            'generations': 1,
            'population_size': 5,
            'cv': 2,
            'scoring': 'f1_macro',
            'max_time_mins': 2,
            'max_eval_time_mins': 1,
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT light'
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info("✅ Teste com NaN concluído com sucesso!")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste com NaN: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import time
    test_tpot_with_nan_data()
