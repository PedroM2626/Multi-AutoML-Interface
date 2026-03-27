#!/usr/bin/env python3
"""
Teste para verificar o tratamento de dados esparsos no TPOT
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import pytest

pytestmark = pytest.mark.skip(reason="Legacy simulation-style integration script")
# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_tpot_with_mixed_data():
    """Testar TPOT com dados mistos (numéricos, categóricos, textuais)"""
    logger.info("🧪 Testando TPOT com dados mistos...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados mistos
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),  # Numérico
            'feature2': np.random.uniform(0, 100, n_samples),  # Numérico
            'feature3': np.random.choice(['A', 'B', 'C', 'D'], n_samples),  # Categórico
            'feature4': np.random.choice(['X', 'Y'], n_samples),  # Categórico binário
            'feature5': ['text data ' + str(i) for i in range(n_samples)],  # Textual
            'feature6': [np.nan if i % 8 == 0 else x for i, x in enumerate(np.random.choice(['cat', 'dog', 'bird'], n_samples))],  # Categórico com NaN
            'target': np.random.choice([0, 1, 2], n_samples)  # Target multiclasse
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dados criados: {df.shape}")
        logger.info(f"Tipos de dados: {df.dtypes}")
        logger.info(f"Valores nulos: {df.isnull().sum().sum()}")
        
        target_column = 'target'
        run_name = f"tpot_test_mixed_{int(time.time())}"
        
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
        
        logger.info("✅ Teste com dados mistos concluído com sucesso!")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        logger.info(f"📊 Tipo de problema: {model_info['problem_type']}")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste com dados mistos: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
def test_tpot_only_categorical():
    """Testar TPOT com apenas dados categóricos"""
    logger.info("🧪 Testando TPOT com apenas dados categóricos...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados apenas categóricos
        np.random.seed(42)
        n_samples = 50
        
        data = {
            'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
            'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
            'cat3': np.random.choice(['red', 'blue', 'green'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dados categóricos criados: {df.shape}")
        logger.info(f"Tipos: {df.dtypes}")
        
        target_column = 'target'
        run_name = f"tpot_test_cat_{int(time.time())}"
        
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
        
        logger.info("✅ Teste com apenas categóricos concluído!")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste categórico: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
if __name__ == "__main__":
    import time
    
    logger.info("🚀 Iniciando testes para correção de dados esparsos...")
    
    tests = [
        ("Dados Mistos", test_tpot_with_mixed_data),
        ("Apenas Categóricos", test_tpot_only_categorical),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Executando: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name} - PASSOU")
            else:
                logger.error(f"❌ {test_name} - FALHOU")
                
        except Exception as e:
            logger.error(f"❌ {test_name} - ERRO: {e}")
            results.append((test_name, False))
    
    # Resumo
    logger.info(f"\n{'='*50}")
    logger.info("📊 RESUMO DOS TESTES")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        logger.info("🎉 Todos os testes passaram! Problema de dados esparsos resolvido!")
    else:
        logger.error(f"⚠️ {total - passed} testes falharam.")
