#!/usr/bin/env python3
"""
Teste para verificar se o erro de timeout foi corrigido no TPOT
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

def test_tpot_no_timeout_param():
    """Testar TPOT sem o parâmetro timeout problemático"""
    logger.info("🧪 Testando TPOT sem parâmetro timeout...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados de teste
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dados de teste criados: {df.shape}")
        
        target_column = 'target'
        run_name = f"tpot_test_timeout_fix_{int(time.time())}"
        
        # Parâmetros que antes causavam erro
        params = {
            'generations': 2,
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
        
        # Treinar modelo (não deve mais dar erro de timeout)
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info("✅ Teste sem timeout concluído com sucesso!")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        logger.info(f"📊 Tipo de problema: {model_info['problem_type']}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste sem timeout: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_tpot_high_dim_without_timeout():
    """Testar TPOT com dados de alta dimensionalidade sem timeout"""
    logger.info("🧪 Testando TPOT com alta dimensionalidade sem timeout...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados que simulam alta dimensionalidade
        np.random.seed(42)
        n_samples = 200
        
        # Features numéricas
        data = {}
        for i in range(50):  # 50 features numéricas
            data[f'num_feature_{i}'] = np.random.randn(n_samples)
        
        # Features categóricas
        for i in range(10):
            data[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
        
        # Features textuais (que geram alta dimensionalidade)
        for i in range(3):
            data[f'text_feature_{i}'] = [
                ' '.join(np.random.choice(['palavra', 'texto', 'dado', 'valor'], 
                                        np.random.randint(3, 10)))
                for _ in range(n_samples)
            ]
        
        # Target
        data['target'] = np.random.choice([0, 1, 2], n_samples)
        
        df = pd.DataFrame(data)
        logger.info(f"Dados de alta dimensionalidade criados: {df.shape}")
        
        target_column = 'target'
        run_name = f"tpot_test_high_dim_no_timeout_{int(time.time())}"
        
        # Parâmetros que devem ser ajustados automaticamente
        params = {
            'generations': 5,  # Deve ser reduzido para <= 2
            'population_size': 15,  # Deve ser reduzido para <= 10
            'cv': 3,  # Deve ser reduzido para <= 3
            'scoring': 'f1_macro',
            'max_time_mins': 3,
            'max_eval_time_mins': 2,  # Deve ser reduzido para <= 1
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT light'
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info("✅ Teste de alta dimensionalidade sem timeout concluído!")
        logger.info(f"📊 Features finais: {model_info['n_features']}")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de alta dimensionalidade: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import time
    
    logger.info("🚀 Iniciando testes para correção do timeout...")
    
    tests = [
        ("Sem Timeout Parameter", test_tpot_no_timeout_param),
        ("Alta Dimensionalidade sem Timeout", test_tpot_high_dim_without_timeout),
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
        logger.info("🎉 Erro de timeout corrigido com sucesso!")
    else:
        logger.error(f"⚠️ {total - passed} testes falharam.")
