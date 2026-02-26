#!/usr/bin/env python3
"""
Script para simular teste H2O AutoML dentro do Docker
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_sample_data():
    """Criar dados de exemplo para teste"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.uniform(0, 100, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    
    # Converter colunas categóricas
    df['feature3'] = df['feature3'].astype('category')
    df['target'] = df['target'].astype('category')
    
    logger.info(f"Dados criados: {df.shape}")
    logger.info(f"Distribuição do target: {df['target'].value_counts()}")
    logger.info(f"Tipos de dados: {df.dtypes}")
    
    return df

def test_h2o_functions():
    """Testar funções H2O individualmente"""
    try:
        # Testar import
        logger.info("Testando import das funções H2O...")
        from h2o_utils import check_java_availability, initialize_h2o, prepare_data_for_h2o
        
        # Testar verificação Java
        logger.info("Testando check_java_availability()...")
        java_available = check_java_availability()
        logger.info(f"Java disponível: {java_available}")
        
        if not java_available:
            logger.error("❌ Java não disponível")
            return False
        
        # Testar inicialização H2O
        logger.info("Testando initialize_h2o()...")
        try:
            h2o_instance = initialize_h2o()
            logger.info(f"✅ H2O inicializado: {type(h2o_instance)}")
        except Exception as e:
            logger.error(f"❌ Erro ao inicializar H2O: {e}")
            return False
        
        # Testar preparação de dados
        logger.info("Testando prepare_data_for_h2o()...")
        df = create_sample_data()
        target = 'target'
        
        try:
            h2o_frame, clean_data = prepare_data_for_h2o(df, target)
            logger.info(f"✅ Dados preparados: {h2o_frame.shape} -> {clean_data.shape}")
            logger.info(f"Tipo H2OFrame: {type(h2o_frame)}")
        except Exception as e:
            logger.error(f"❌ Erro ao preparar dados: {e}")
            return False
        
        # Limpar
        try:
            from h2o_utils import cleanup_h2o
            cleanup_h2o()
            logger.info("✅ H2O cleanup concluído")
        except Exception as e:
            logger.warning(f"Aviso no cleanup: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro geral: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_h2o_training_minimal():
    """Testar treinamento H2O mínimo"""
    try:
        from h2o_utils import train_h2o_model
        
        # Criar dados pequenos para teste rápido
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'target': np.random.choice([0, 1], n_samples)
        }
        
        df = pd.DataFrame(data)
        logger.info(f"Dados de teste: {df.shape}")
        
        # Parâmetros mínimos
        run_name = f"minimal_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        params = {
            'max_runtime_secs': 30,  # 30 segundos
            'max_models': 2,         # Apenas 2 modelos
            'nfolds': 2,             # 2 folds
            'balance_classes': True,
            'seed': 42,
            'sort_metric': 'AUTO',
            'exclude_algos': ['DeepLearning', 'GLM']  # Excluir algoritmos lentos
        }
        
        logger.info(f"Iniciando treinamento mínimo com parâmetros: {params}")
        
        # Treinar
        automl, run_id = train_h2o_model(df, 'target', run_name, **params)
        
        logger.info(f"✅ Treinamento concluído!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Tipo AutoML: {type(automl)}")
        
        # Verificar leader
        if hasattr(automl, 'leader'):
            leader = automl.leader
            logger.info(f"Tipo leader: {type(leader)}")
            
            if leader is None:
                logger.error("❌ Leader é None!")
                return False
            else:
                logger.info("✅ Leader não é None!")
                logger.info(f"Model ID: {leader.model_id if hasattr(leader, 'model_id') else 'N/A'}")
                
                # Testar predição básica
                try:
                    test_data = df.head(5).drop('target', axis=1)
                    from h2o_utils import predict_with_h2o
                    predictions = predict_with_h2o(leader, test_data)
                    logger.info(f"✅ Predição teste: {predictions}")
                except Exception as e:
                    logger.error(f"❌ Erro na predição: {e}")
                    return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Função principal"""
    logger.info("🐳 Teste H2O AutoML (Ambiente Docker)")
    logger.info("=" * 50)
    
    # Teste 1: Funções básicas
    logger.info("\n🔧 TESTE 1: FUNÇÕES BÁSICAS")
    logger.info("-" * 30)
    
    basic_success = test_h2o_functions()
    
    if not basic_success:
        logger.error("❌ Falha nas funções básicas")
        return
    
    # Teste 2: Treinamento mínimo
    logger.info("\n🚀 TESTE 2: TREINAMENTO MÍNIMO")
    logger.info("-" * 30)
    
    training_success = test_h2o_training_minimal()
    
    # Resumo
    logger.info("\n📋 RESUMO DOS TESTES")
    logger.info("-" * 30)
    logger.info(f"Funções básicas: {'✅' if basic_success else '❌'}")
    logger.info(f"Treinamento: {'✅' if training_success else '❌'}")
    
    if basic_success and training_success:
        logger.info("\n🎉 H2O AutoML está funcionando perfeitamente!")
        logger.info("A interface Streamlit deve funcionar corretamente.")
    else:
        logger.info("\n❌ H2O AutoML precisa de correções.")

if __name__ == "__main__":
    main()
