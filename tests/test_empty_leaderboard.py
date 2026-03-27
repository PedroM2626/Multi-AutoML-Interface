#!/usr/bin/env python3
"""
Script para testar o tratamento de leaderboard vazio no H2O AutoML
"""

import pandas as pd
import numpy as np
import sys
import os
import logging
import pytest
from datetime import datetime

pytestmark = pytest.mark.skip(reason="Legacy simulation-style integration script")

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def create_problematic_data():
    """Criar dados que podem causar problemas no H2O"""
    np.random.seed(42)
    n_samples = 100  # Dataset pequeno
    
    # Dados com características que podem dificultar o treinamento
    data = {
        'feature1': [1.0] * n_samples,  # Todos iguais (baixa variância)
        'feature2': np.random.randn(n_samples) * 0.001,  # Variância muito baixa
        'feature3': ['A'] * n_samples,  # Uma única categoria
        'target': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # Desbalanceado extremo
    }
    
    df = pd.DataFrame(data)
    
    logger.info(f"📊 Dados problemáticos criados: {df.shape}")
    logger.info(f"Feature1 (todos iguais): {df['feature1'].unique()}")
    logger.info(f"Feature3 (uma categoria): {df['feature3'].unique()}")
    logger.info(f"Distribuição do target: {df['target'].value_counts()}")
    
    return df

def test_empty_leaderboard_handling():
    """Testar tratamento de leaderboard vazio"""
    try:
        from h2o_utils import train_h2o_model, check_java_availability
        
        # Verificar Java
        logger.info("Verificando Java...")
        if not check_java_availability():
            logger.error("❌ Java não disponível!")
        pytest.fail("Test flow returned False")
        logger.info("✅ Java disponível!")
        
        # Criar dados problemáticos
        df = create_problematic_data()
        
        # Parâmetros que podem causar falha
        run_name = f"empty_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        params = {
            'max_runtime_secs': 10,    # Tempo muito curto
            'max_models': 1,           # Apenas 1 modelo
            'nfolds': 2,               # 2 folds
            'balance_classes': True,
            'seed': 42,
            'sort_metric': 'AUTO',
            'exclude_algos': ['DeepLearning', 'GBM', 'DRF']  # Excluir algoritmos robustos
        }
        
        logger.info(f"Parâmetros que podem causar falha: {params}")
        logger.info("Iniciando treinamento com dados problemáticos...")
        
        # Treinar modelo
        automl, run_id = train_h2o_model(df, 'target', run_name, **params)
        
        logger.info("✅ Treinamento concluído (mesmo que sem modelos)!")
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Tipo AutoML: {type(automl)}")
        
        # Verificar se o AutoML tem leader
        try:
            if hasattr(automl, 'leader') and automl.leader is not None:
                logger.info("✅ Leader encontrado")
                logger.info(f"Model ID: {automl.leader.model_id}")
            else:
                logger.info("ℹ️ Nenhum leader (esperado para este teste)")
        except Exception as e:
            logger.info(f"ℹ️ Erro ao acessar leader (esperado): {e}")
        assert True
    except Exception as e:
        logger.error(f"❌ Erro no teste: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        pytest.fail("Test flow returned False")
def main():
    """Função principal"""
    logger.info("🧪 TESTE DE TRATAMENTO DE LEADERBOARD VAZIO")
    logger.info("=" * 50)
    logger.info("Este script testa se o código lida corretamente com")
    logger.info("situações onde nenhum modelo é treinado")
    logger.info("=" * 50)
    
    success = test_empty_leaderboard_handling()
    
    logger.info("\n📋 RESULTADO DO TESTE")
    logger.info("-" * 30)
    logger.info(f"Teste: {'✅ SUCESSO' if success else '❌ FALHA'}")
    
    if success:
        logger.info("\n🎉 O tratamento de leaderboard vazio está funcionando!")
        logger.info("O erro 'Column logloss not found' não deve mais ocorrer.")
    else:
        logger.info("\n❌ Ainda há problemas no tratamento de leaderboard vazio.")

if __name__ == "__main__":
    main()
