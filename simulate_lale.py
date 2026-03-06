import pandas as pd
import threading
import time
import os
from src.lale_utils import run_lale_experiment
import mlflow

def simulate_lale():
    print("Inciando simulacao do Lale...")
    
    # Criando um dataset de mentira (Dummy Dataset)
    print("Gerando dataset Dummy...")
    df = pd.DataFrame({
        "idade": [22, 25, 47, 52, 46, 36, 21, 28, 30, 41] * 10,
        "salario": [3000, 4500, 8000, 10000, 8500, 5000, 2000, 3500, 4000, 7000] * 10,
        "comprou": [0, 0, 1, 1, 1, 0, 0, 0, 0, 1] * 10
    })
    
    # MLflow tracking
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    
    # Chamando o wrapper da mesma forma que o app.py chama
    tempo_limite = 60 # max_evals associado (pequeno para teste)
    
    start = time.time()
    try:
        resultado = run_lale_experiment(
            train_df=df,
            target_col="comprou",
            run_name="simulacao_lale_teste",
            log_queue=None,
            time_limit=tempo_limite,
            val_df=None,
            stop_event=threading.Event()
        )
        end = time.time()
        
        print("\nSimulacao concluida com sucesso!")
        print(f"Tempo: {end - start:.2f} segundos")
        print("Resumo dos Resultados:")
        print(f"Tipo: {resultado.get('type')}")
        print(f"Run ID MLflow: {resultado.get('run_id')}")
        print(f"Métricas: {resultado.get('metrics')}")
    except Exception as e:
        print(f"\nErro durante a simulacao: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_lale()
