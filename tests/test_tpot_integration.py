#!/usr/bin/env python3
"""
Teste de integração do TPOT AutoML com o Multi-AutoML Interface
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import pytest
import time
from datetime import datetime

pytestmark = pytest.mark.skip(reason="Legacy simulation-style integration script")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_test_data():
    """Criar dados de teste para TPOT"""
    np.random.seed(42)
    n_samples = 500
    
    # Dados de classificação
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    return df

def test_tpot_classification():
    """Testar TPOT para classificação"""
    logger.info("🧪 Testando TPOT para classificação...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados de teste
        df = create_test_data()
        target_column = 'target'
        run_name = f"tpot_test_classification_{int(time.time())}"
        
        # Parâmetros para teste rápido
        params = {
            'generations': 2,
            'population_size': 10,
            'cv': 3,
            'scoring': 'f1_macro',
            'max_time_mins': 5,  # 5 minutos para teste
            'max_eval_time_mins': 2,
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT light'
        }
        
        logger.info(f"Parâmetros: {params}")
        
        # Treinar modelo
        start_time = time.time()
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        training_time = time.time() - start_time
        
        logger.info(f"✅ Treinamento concluído em {training_time:.2f} segundos")
        logger.info(f"📊 Run ID: {run_id}")
        logger.info(f"🎯 Tipo de problema: {model_info['problem_type']}")
        logger.info(f"🧬 Pipeline: {str(tpot.fitted_pipeline_)}")
        
        # Verificar métricas
        if model_info['problem_type'] == 'classification':
            assert 'accuracy' in model_info, "Accuracy não encontrada nas métricas"
            assert 'f1_macro' in model_info, "F1 macro não encontrada nas métricas"
            logger.info(f"📈 Accuracy: {model_info['accuracy']:.4f}")
            logger.info(f"📈 F1 Macro: {model_info['f1_macro']:.4f}")
        
        # Verificar se arquivos foram criados
        import os
        assert os.path.exists("tpot_models"), "Pasta tpot_models não criada"
        assert os.path.exists(f"tpot_models/best_pipeline_{run_name}.py"), "Pipeline não exportado"
        assert os.path.exists(f"tpot_models/model_info_{run_name}.txt"), "Model info não salvo"
        
        logger.info("✅ Teste de classificação TPOT concluído com sucesso!")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste de classificação TPOT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
def test_tpot_regression():
    """Testar TPOT para regressão"""
    logger.info("🧪 Testando TPOT para regressão...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados de regressão
        np.random.seed(42)
        n_samples = 300
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.uniform(0, 100, n_samples),
            'target': np.random.randn(n_samples) * 10 + 5
        }
        
        df = pd.DataFrame(data)
        target_column = 'target'
        run_name = f"tpot_test_regression_{int(time.time())}"
        
        # Parâmetros para teste rápido
        params = {
            'generations': 2,
            'population_size': 10,
            'cv': 3,
            'scoring': 'neg_mean_squared_error',
            'max_time_mins': 5,
            'max_eval_time_mins': 2,
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT light'
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info(f"✅ Treinamento de regressão concluído")
        logger.info(f"🎯 Tipo de problema: {model_info['problem_type']}")
        
        # Verificar métricas de regressão
        assert model_info['problem_type'] == 'regression', "Tipo de problema incorreto"
        assert 'rmse' in model_info, "RMSE não encontrado nas métricas"
        assert 'r2' in model_info, "R² não encontrado nas métricas"
        logger.info(f"📈 RMSE: {model_info['rmse']:.4f}")
        logger.info(f"📈 R²: {model_info['r2']:.4f}")
        
        logger.info("✅ Teste de regressão TPOT concluído com sucesso!")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste de regressão TPOT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
def test_tpot_with_text_data():
    """Testar TPOT com dados textuais"""
    logger.info("🧪 Testando TPOT com dados textuais...")
    
    try:
        from tpot_utils import train_tpot_model
        
        # Criar dados com texto
        np.random.seed(42)
        n_samples = 200
        data = {
            'text_feature': [
                'positive review' if i % 2 == 0 else 'negative review' 
                for i in range(n_samples)
            ],
            'numeric_feature': np.random.randn(n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        target_column = 'target'
        run_name = f"tpot_test_text_{int(time.time())}"
        
        # Parâmetros para teste rápido
        params = {
            'generations': 2,
            'population_size': 10,
            'cv': 3,
            'scoring': 'f1_macro',
            'max_time_mins': 5,
            'max_eval_time_mins': 2,
            'random_state': 42,
            'verbosity': 1,
            'n_jobs': 1,
            'config_dict': 'TPOT sparse'  # Config para dados esparsos
        }
        
        # Treinar modelo
        tpot, pipeline, run_id, model_info = train_tpot_model(
            df, target_column, run_name, **params
        )
        
        logger.info(f"✅ Treinamento com texto concluído")
        logger.info(f"📝 Colunas de texto: {model_info.get('text_columns', [])}")
        logger.info(f"🧬 Pipeline com TF-IDF: {str(tpot.fitted_pipeline_)}")
        
        logger.info("✅ Teste com dados textuais concluído com sucesso!")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste com dados textuais: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
def test_problem_type_detection():
    """Testar detecção automática de tipo de problema"""
    logger.info("🧪 Testando detecção de tipo de problema...")
    
    try:
        from tpot_utils import detect_problem_type
        
        # Testar classificação
        y_class = pd.Series([0, 1, 0, 1, 1])
        problem_type = detect_problem_type(y_class)
        assert problem_type == 'classification', f"Esperado classification, got {problem_type}"
        
        # Testar regressão
        y_reg = pd.Series([1.5, 2.7, 3.1, 4.8, 5.2])
        problem_type = detect_problem_type(y_reg)
        assert problem_type == 'regression', f"Esperado regression, got {problem_type}"
        
        # Testar classificação com strings
        y_str = pd.Series(['A', 'B', 'A', 'C', 'B'])
        problem_type = detect_problem_type(y_str)
        assert problem_type == 'classification', f"Esperado classification para strings, got {problem_type}"
        
        logger.info("✅ Detecção de tipo de problema funcionando corretamente!")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro na detecção de tipo de problema: {e}")
        pytest.fail("Test flow returned False")
def test_feature_pipeline():
    """Testar criação de pipeline de features"""
    logger.info("🧪 Testando pipeline de features...")
    
    try:
        from tpot_utils import create_feature_pipeline
        
        # Criar dados mistos
        df = pd.DataFrame({
            'text_col': ['hello world', 'test data', 'more text'],
            'num_col1': [1.0, 2.0, 3.0],
            'num_col2': [4, 5, 6],
            'cat_col': ['A', 'B', 'A'],
            'target': [0, 1, 0]
        })
        
        preprocessor, text_cols, cat_cols, num_cols = create_feature_pipeline(
            df, 'target', text_columns=['text_col']
        )
        
        assert text_cols == ['text_col'], f"Text cols incorreto: {text_cols}"
        assert 'cat_col' in cat_cols, "Coluna categórica não detectada"
        assert 'num_col1' in num_cols and 'num_col2' in num_cols, "Colunas numéricas não detectadas"
        
        logger.info(f"✅ Pipeline criado com sucesso:")
        logger.info(f"   Text columns: {text_cols}")
        logger.info(f"   Categorical columns: {cat_cols}")
        logger.info(f"   Numerical columns: {num_cols}")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no pipeline de features: {e}")
        import traceback
        logger.error(traceback.format_exc())
        pytest.fail("Test flow returned False")
def run_all_tests():
    """Executar todos os testes"""
    logger.info("🚀 Iniciando testes de integração TPOT...")
    
    tests = [
        ("Detecção de Tipo de Problema", test_problem_type_detection),
        ("Pipeline de Features", test_feature_pipeline),
        ("Classificação TPOT", test_tpot_classification),
        ("Regressão TPOT", test_tpot_regression),
        ("Dados Textuais TPOT", test_tpot_with_text_data),
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
    
    # Resumo final
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
        logger.info("🎉 Todos os testes passaram! TPOT está integrado com sucesso!")
        assert True
    else:
        logger.error(f"⚠️ {total - passed} testes falharam. Verifique os erros acima.")
        pytest.fail("Test flow returned False")
if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ Erro geral nos testes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)
