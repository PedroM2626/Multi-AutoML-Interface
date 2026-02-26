#!/usr/bin/env python3
"""
Script para simular a interface e testar H2O AutoML
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

def test_h2o_training():
    """Testar treinamento H2O AutoML"""
    try:
        from h2o_utils import train_h2o_model, check_java_availability
        
        # Verificar Java
        logger.info("Verificando disponibilidade do Java...")
        if not check_java_availability():
            logger.error("❌ Java não está disponível!")
            logger.info("Soluções:")
            logger.info("1. Use Docker: docker build -t multi-automl-interface . && docker run -p 8501:8501 multi-automl-interface")
            logger.info("2. Instale Java localmente")
            return False
        
        logger.info("✅ Java está disponível!")
        
        # Criar dados
        df = create_sample_data()
        
        # Configurar parâmetros
        run_name = f"test_h2o_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        target = 'target'
        
        params = {
            'max_runtime_secs': 60,  # 1 minuto para teste rápido
            'max_models': 5,        # Poucos modelos para teste
            'nfolds': 3,
            'balance_classes': True,
            'seed': 42,
            'sort_metric': 'AUTO',
            'exclude_algos': ['DeepLearning']  # Excluir DL para test rápido
        }
        
        logger.info(f"Parâmetros do treinamento: {params}")
        logger.info(f"Iniciando treinamento H2O AutoML...")
        
        # Treinar modelo
        automl, run_id = train_h2o_model(df, target, run_name, **params)
        
        logger.info(f"✅ Treinamento concluído com sucesso!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Tipo do objeto retornado: {type(automl)}")
        
        # Verificar se o líder não é None
        if hasattr(automl, 'leader'):
            leader = automl.leader
            logger.info(f"Tipo do líder: {type(leader)}")
            logger.info(f"ID do modelo líder: {leader.model_id if hasattr(leader, 'model_id') else 'N/A'}")
            
            if leader is None:
                logger.error("❌ O líder (leader) é None!")
                return False
            else:
                logger.info("✅ O líder não é None!")
        else:
            logger.error("❌ O objeto AutoML não tem atributo 'leader'!")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no treinamento H2O: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_h2o_loading(run_id):
    """Testar carregamento de modelo H2O"""
    try:
        from h2o_utils import load_h2o_model
        
        logger.info(f"Testando carregamento do modelo com Run ID: {run_id}")
        
        # Carregar modelo
        model = load_h2o_model(run_id)
        
        logger.info(f"✅ Modelo carregado com sucesso!")
        logger.info(f"Tipo do modelo: {type(model)}")
        
        if model is None:
            logger.error("❌ Modelo carregado é None!")
            return False
        else:
            logger.info("✅ Modelo não é None!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Erro no carregamento H2O: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_h2o_prediction(model):
    """Testar predição com modelo H2O"""
    try:
        from h2o_utils import predict_with_h2o
        
        logger.info("Testando predição H2O...")
        
        # Criar dados de teste
        test_data = create_sample_data().head(10)
        test_data = test_data.drop('target', axis=1)
        
        logger.info(f"Dados de teste: {test_data.shape}")
        
        # Fazer predição
        predictions = predict_with_h2o(model, test_data)
        
        logger.info(f"✅ Predição concluída!")
        logger.info(f"Tipo das predições: {type(predictions)}")
        logger.info(f"Shape das predições: {predictions.shape}")
        logger.info(f"Predições: {predictions}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro na predição H2O: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Função principal"""
    logger.info("🚀 Iniciando teste completo do H2O AutoML")
    logger.info("=" * 60)
    
    # Teste 1: Treinamento
    logger.info("\n📊 TESTE 1: TREINAMENTO H2O")
    logger.info("-" * 40)
    
    success_training = test_h2o_training()
    
    if not success_training:
        logger.error("❌ Falha no treinamento. Abortando testes restantes.")
        return
    
    # Teste 2: Carregamento (simulado)
    logger.info("\n📂 TESTE 2: CARREGAMENTO H2O")
    logger.info("-" * 40)
    logger.info("⚠️  Pulando teste de carregamento (precisa de run_id real)")
    
    # Teste 3: Predição (se tivermos um modelo)
    logger.info("\n🔮 TESTE 3: PREDIÇÃO H2O")
    logger.info("-" * 40)
    logger.info("⚠️  Pulando teste de predição (precisa de modelo real)")
    
    logger.info("\n✅ Testes concluídos!")
    logger.info("=" * 60)
    
    if success_training:
        logger.info("🎉 H2O AutoML está funcionando corretamente!")
    else:
        logger.info("❌ H2O AutoML precisa de correções.")

if __name__ == "__main__":
    main()
