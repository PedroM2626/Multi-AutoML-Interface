#!/usr/bin/env python3
"""
Teste para verificar o tratamento de dados grandes e alta dimensionalidade no TPOT
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from sklearn.datasets import make_classification, make_regression
from sklearn.feature_extraction.text import TfidfVectorizer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_high_dimensional_data():
    """Criar dados de alta dimensionalidade (65k+ features)"""
    logger.info("🧪 Criando dados de alta dimensionalidade...")
    
    # Criar dados base
    n_samples = 1000
    n_features = 100  # Features numéricas base
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=20,
        n_redundant=10,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Converter para DataFrame
    df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    # Adicionar features categóricas
    for i in range(10):
        df[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
    
    # Adicionar features textuais (que vão gerar alta dimensionalidade com TF-IDF)
    for i in range(5):
        df[f'text_feature_{i}'] = [
            ' '.join(np.random.choice(['word1', 'word2', 'word3', 'word4', 'word5'], 
                                    np.random.randint(5, 20)))
            for _ in range(n_samples)
        ]
    
    logger.info(f"Dados criados: {df.shape}")
    logger.info(f"Features textuais: 5 (vão gerar alta dimensionalidade)")
    
    return df

def create_large_dataset():
    """Criar dataset grande (50k+ amostras)"""
    logger.info("🧪 Criando dataset grande...")
    
    # Criar dados base
    n_samples = 50000
    n_features = 50
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Converter para DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    # Adicionar algumas features categóricas
    for i in range(5):
        df[f'cat_feature_{i}'] = np.random.choice(['A', 'B', 'C'], n_samples)
    
    logger.info(f"Dataset grande criado: {df.shape}")
    
    return df

def test_tpot_high_dimensional():
    """Testar TPOT com dados de alta dimensionalidade"""
    logger.info("🧪 Testando TPOT com alta dimensionalidade...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados de alta dimensionalidade
        df = create_high_dimensional_data()
        
        target_column = 'target'
        run_name = f"tpot_test_high_dim_{int(time.time())}"
        
        # Parâmetros para teste (devem ser ajustados automaticamente)
        params = {
            'generations': 5,
            'population_size': 20,
            'cv': 3,
            'scoring': 'f1_macro',
            'max_time_mins': 5,  # Tempo curto para teste
            'max_eval_time_mins': 2,
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT sparse'
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info("✅ Teste de alta dimensionalidade concluído!")
        logger.info(f"📊 Features finais: {model_info['n_features']}")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de alta dimensionalidade: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_tpot_large_dataset():
    """Testar TPOT com dataset grande"""
    logger.info("🧪 Testando TPOT com dataset grande...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dataset grande
        df = create_large_dataset()
        
        target_column = 'target'
        run_name = f"tpot_test_large_{int(time.time())}"
        
        # Parâmetros para teste (devem ser ajustados automaticamente)
        params = {
            'generations': 3,
            'population_size': 10,
            'cv': 5,
            'scoring': 'f1_macro',
            'max_time_mins': 3,  # Tempo curto para teste
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
        
        logger.info("✅ Teste de dataset grande concluído!")
        logger.info(f"📊 Amostras: {model_info['n_samples']}")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de dataset grande: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_parameter_adjustment():
    """Testar se os parâmetros são ajustados corretamente"""
    logger.info("🧪 Testando ajuste automático de parâmetros...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados extremos (alta dimensionalidade + muitas amostras)
        df = create_high_dimensional_data()
        
        target_column = 'target'
        run_name = f"tpot_test_adjust_{int(time.time())}"
        
        # Parâmetros que devem ser ajustados
        params = {
            'generations': 10,  # Deve ser reduzido para <= 2
            'population_size': 50,  # Deve ser reduzido para <= 10
            'cv': 5,  # Deve ser reduzido para <= 3
            'scoring': 'f1_macro',
            'max_time_mins': 10,
            'max_eval_time_mins': 5,  # Deve ser reduzido para <= 2
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT sparse'  # Deve ser mantido
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        # Verificar se os parâmetros foram ajustados
        logger.info("✅ Teste de ajuste de parâmetros concluído!")
        logger.info(f"📊 Parâmetros ajustados automaticamente")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no teste de ajuste de parâmetros: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    import time
    
    logger.info("🚀 Iniciando testes para dados grandes e alta dimensionalidade...")
    
    tests = [
        ("Alta Dimensionalidade", test_tpot_high_dimensional),
        ("Dataset Grande", test_tpot_large_dataset),
        ("Ajuste de Parâmetros", test_parameter_adjustment),
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
        logger.info("🎉 Todos os testes passaram! TPOT pronto para dados grandes!")
    else:
        logger.error(f"⚠️ {total - passed} testes falharam.")
