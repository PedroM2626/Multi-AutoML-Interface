import pandas as pd
import os
from src.flaml_utils import train_flaml_model
import mlflow

def simulate_flaml_training():
    dataset_path = r"c:\Users\pedro\Downloads\test\processed_train.csv"
    
    if not os.path.exists(dataset_path):
        print(f"Erro: Dataset não encontrado em {dataset_path}")
        return

    print(f"Carregando dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Identificando o target
    target = 'target' # Ajustado conforme o dataset esperado
    if target not in df.columns:
        target = df.columns[-1]
    
    print(f"Coluna alvo identificada: {target}")
    
    run_name = "simulated_flaml_run"
    time_budget = 10  # Reduzido para teste rápido
    task = 'classification'
    metric = 'accuracy' # Métrica explícita
    estimator_list = ['lgbm'] # Apenas um estimador

    print("\n--- Iniciando Simulação de Treinamento FLAML ---")
    try:
        automl, run_id = train_flaml_model(
            train_data=df,
            target=target,
            run_name=run_name,
            time_budget=time_budget,
            task=task,
            metric=metric,
            estimator_list=estimator_list
        )
        
        print("\n✅ Treinamento concluído com sucesso!")
        print(f"Run ID: {run_id}")
        print(f"Melhor Estimador: {automl.best_estimator}")
        print(f"Melhor Loss: {automl.best_loss}")
        
        # Verificar se o modelo foi salvo nos artefatos do MLflow
        run = mlflow.get_run(run_id)
        print(f"Status da Run no MLflow: {run.info.status}")
        
    except Exception as e:
        print(f"\n❌ Erro durante a simulação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simulate_flaml_training()
