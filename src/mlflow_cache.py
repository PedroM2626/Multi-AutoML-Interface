import mlflow
import pandas as pd
import time
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class MLflowCache:
    """Cache para otimizar carregamento de dados do MLflow"""
    
    def __init__(self, ttl: int = 300):  # TTL de 5 minutos
        self._cache = {}
        self._timestamps = {}
        self.ttl = ttl
    
    def _is_expired(self, key: str) -> bool:
        """Verifica se o cache expirou"""
        if key not in self._timestamps:
            return True
        return time.time() - self._timestamps[key] > self.ttl
    
    def _set_cache(self, key: str, value):
        """Define valor no cache"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def get_cached_all_runs(self, experiment_name: str) -> pd.DataFrame:
        """Obtém todas as runs com cache"""
        cache_key = f"all_runs_{experiment_name}"
        
        if not self._is_expired(cache_key) and cache_key in self._cache:
            logger.info(f"Usando cache para experimento {experiment_name}")
            return self._cache[cache_key]
        
        try:
            # Obter experimento
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                return pd.DataFrame()
            
            # Buscar runs
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            
            # Cache do resultado
            self._set_cache(cache_key, runs)
            logger.info(f"Cache atualizado para experimento {experiment_name} ({len(runs)} runs)")
            
            return runs
            
        except Exception as e:
            logger.error(f"Erro ao buscar runs do experimento {experiment_name}: {e}")
            return pd.DataFrame()
    
    def get_cached_experiment(self, experiment_name: str):
        """Obtém experimento com cache"""
        cache_key = f"experiment_{experiment_name}"
        
        if not self._is_expired(cache_key) and cache_key in self._cache:
            logger.info(f"Usando cache para experimento {experiment_name}")
            return self._cache[cache_key]
        
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self._set_cache(cache_key, experiment)
            return experiment
            
        except Exception as e:
            logger.error(f"Erro ao buscar experimento {experiment_name}: {e}")
            return None
    
    def clear_cache(self):
        """Limpa todo o cache"""
        self._cache.clear()
        self._timestamps.clear()
        logger.info("Cache limpo")
    
    def clear_experiment_cache(self, experiment_name: str):
        """Limpa cache de um experimento específico"""
        keys_to_remove = [key for key in self._cache.keys() if experiment_name in key]
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._timestamps.pop(key, None)
        logger.info(f"Cache do experimento {experiment_name} limpo")

# Instância global do cache
mlflow_cache = MLflowCache()

@lru_cache(maxsize=128)
def get_cached_experiment_list():
    """Obtém lista de experimentos com cache"""
    try:
        experiments = mlflow.search_experiments()
        return [exp.name for exp in experiments]
    except Exception as e:
        logger.error(f"Erro ao buscar lista de experimentos: {e}")
        return ["AutoGluon_Experiments", "FLAML_Experiments", "H2O_Experiments"]
