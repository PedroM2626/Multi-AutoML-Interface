import os
import pandas as pd
import mlflow
import shutil
import logging
import time
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.mlflow_utils import safe_set_experiment

logger = logging.getLogger(__name__)

def check_java_availability():
    """Verifica se Java está disponível no sistema"""
    try:
        import subprocess
        import os
        
        # Tentar encontrar Java no PATH
        result = subprocess.run(['java', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
        
        # Se não encontrar no PATH, tentar caminhos comuns no Windows
        java_paths = [
            r"C:\Program Files\Eclipse Adoptium\jdk-11.0.30.7-hotspot\bin\java.exe",
            r"C:\Program Files\Eclipse Adoptium\jdk-11.0.23.9-hotspot\bin\java.exe",
            r"C:\Program Files\Java\jdk-11\bin\java.exe",
            r"C:\Program Files\Java\jdk-17\bin\java.exe",
        ]
        
        for java_path in java_paths:
            if os.path.exists(java_path):
                result = subprocess.run([java_path, '-version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return True
        
        return False
        
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False

def initialize_h2o():
    """Inicializa o cluster H2O com verificação de Java"""
    if not check_java_availability():
        raise RuntimeError(
            "Java não está instalado no sistema. H2O AutoML requer Java para funcionar.\n\n"
            "Opções:\n"
            "1. Instalar Java localmente (JRE/JDK)\n"
            "2. Usar Docker: docker build -t multi-automl-interface . && docker run -p 8501:8501 multi-automl-interface\n"
            "3. Usar AutoGluon ou FLAML como alternativas (não requerem Java)\n"
            "\nPara instalar Java no Windows:\n"
            "- Baixe em: https://adoptium.net/\n"
            "- Ou use: winget install EclipseAdoptium.Temurin.11.JDK"
        )
    
    try:
        import h2o
        h2o.init(max_mem_size="4G", nthreads=-1)
        logger.info("Cluster H2O inicializado com sucesso")
        return h2o
    except Exception as e:
        logger.error(f"Erro ao inicializar H2O: {e}")
        raise

def cleanup_h2o():
    """Finaliza o cluster H2O"""
    try:
        import h2o
        h2o.cluster().shutdown()
        logger.info("Cluster H2O finalizado")
    except Exception as e:
        logger.warning(f"Erro ao finalizar H2O: {e}")

def prepare_data_for_h2o(train_data: pd.DataFrame, target: str):
    """Prepara dados para o H2O AutoML"""
    import h2o
    
    # Remover valores nulos
    train_data_clean = train_data.dropna(subset=[target])
    
    # Para dados textuais, criar features numéricas básicas
    if train_data_clean.select_dtypes(include=['object']).shape[1] > 0:
        logger.info("Detectadas colunas textuais, criando features numéricas básicas...")
        
        # Para cada coluna textual, criar features básicas
        for col in train_data_clean.select_dtypes(include=['object']).columns:
            if col != target:
                # Comprimento do texto
                train_data_clean[f'{col}_length'] = train_data_clean[col].astype(str).str.len()
                # Número de palavras
                train_data_clean[f'{col}_word_count'] = train_data_clean[col].astype(str).str.split().str.len()
                
        # Remover colunas textuais exceto o target
        text_cols = train_data_clean.select_dtypes(include=['object']).columns
        text_cols = [col for col in text_cols if col != target]
        train_data_clean = train_data_clean.drop(columns=text_cols)
    
    # Converter para H2OFrame
    h2o_frame = h2o.H2OFrame(train_data_clean)
    
    # Converter target para fator (categórico) se for classificação
    if train_data_clean[target].dtype == 'object' or train_data_clean[target].nunique() < 20:
        h2o_frame[target] = h2o_frame[target].asfactor()
    
    return h2o_frame, train_data_clean

def train_h2o_model(train_data: pd.DataFrame, target: str, run_name: str, 
                   max_runtime_secs: int = 300, max_models: int = 10, 
                   nfolds: int = 3, balance_classes: bool = True, seed: int = 42,
                   sort_metric: str = "AUTO", exclude_algos: list = None):
    """
    Treina modelo H2O AutoML e registra no MLflow
    """
    import h2o
    from h2o.automl import H2OAutoML
    
    safe_set_experiment("H2O_Experiments")
    logging.info(f"Iniciando treinamento H2O AutoML para a run: {run_name}")
    
    # Inicializar H2O
    h2o_instance = initialize_h2o()
    
    try:
        with mlflow.start_run(run_name=run_name) as run:
            # Preparar dados
            h2o_frame, clean_data = prepare_data_for_h2o(train_data, target)
            
            # Log parâmetros
            mlflow.log_param("target", target)
            mlflow.log_param("max_runtime_secs", max_runtime_secs)
            mlflow.log_param("max_models", max_models)
            mlflow.log_param("nfolds", nfolds)
            mlflow.log_param("balance_classes", balance_classes)
            mlflow.log_param("seed", seed)
            mlflow.log_param("sort_metric", sort_metric)
            mlflow.log_param("model_type", "h2o_automl")
            if exclude_algos:
                mlflow.log_param("exclude_algos", exclude_algos)
            
            # Definir features (todas exceto target)
            features = [col for col in clean_data.columns if col != target]
            mlflow.log_param("features", features)
            
            # Configurar AutoML
            aml = H2OAutoML(
                max_runtime_secs=max_runtime_secs,
                max_models=max_models,
                seed=seed,
                nfolds=nfolds,
                balance_classes=balance_classes,
                keep_cross_validation_predictions=True,
                keep_cross_validation_models=False,
                verbosity='info',
                sort_metric=sort_metric,
                exclude_algos=exclude_algos or []
            )
            
            # Treinar modelo
            logger.info("Iniciando treinamento H2O AutoML...")
            start_time = time.time()
            aml.train(x=features, y=target, training_frame=h2o_frame)
            training_duration = time.time() - start_time
            
            logger.info(f"Treinamento concluído em {training_duration:.2f} segundos")
            
            # Obter o leaderboard
            leaderboard = aml.leaderboard
            
            # Verificar se o leaderboard está vazio
            if leaderboard.nrow == 0:
                logger.warning("⚠️ Nenhum modelo foi treinado. O leaderboard está vazio.")
                logger.warning("Isso pode acontecer se:")
                logger.warning("1. O tempo máximo for muito curto")
                logger.warning("2. Os dados não forem adequados para os algoritmos")
                logger.warning("3. Houver problemas com os dados")
                
                # Logar métricas básicas mesmo sem modelos
                mlflow.log_metric("total_models_trained", 0)
                mlflow.log_metric("training_duration", training_duration)
                mlflow.log_metric("best_model_score", 0.0)
                
                # Retornar o AutoML mesmo sem modelos
                return aml, run.info.run_id
            
            logger.info("\nTop 5 modelos:")
            print(leaderboard.head(5))
            
            # Salvar leaderboard como métrica com tratamento seguro
            try:
                # Verificar colunas disponíveis no leaderboard
                leaderboard_df = None
                try:
                    leaderboard_df = leaderboard.as_data_frame()
                    logger.info(f"Colunas disponíveis: {list(leaderboard_df.columns)}")
                except Exception as e:
                    logger.warning(f"Não foi possível converter leaderboard para DataFrame: {e}")
                
                # Tentar obter a melhor métrica disponível
                best_model_score = 0.0
                if leaderboard_df is not None and len(leaderboard_df) > 0:
                    # Procurar métricas em ordem de preferência
                    for metric in ['auc', 'logloss', 'rmse', 'mae', 'r2']:
                        if metric in leaderboard_df.columns:
                            best_model_score = leaderboard_df.iloc[0][metric]
                            logger.info(f"Usando métrica '{metric}': {best_model_score}")
                            break
                    
                    mlflow.log_metric("total_models_trained", len(leaderboard_df))
                else:
                    # Fallback: usar o primeiro valor do leaderboard H2O
                    try:
                        available_columns = leaderboard.columns
                        logger.info(f"Colunas H2O disponíveis: {available_columns}")
                        
                        # Tentar acessar primeira linha, primeira coluna
                        if len(available_columns) > 0:
                            first_col = available_columns[0]
                            best_model_score = leaderboard[0, first_col]
                            logger.info(f"Usando primeira coluna disponível '{first_col}': {best_model_score}")
                        
                        mlflow.log_metric("total_models_trained", leaderboard.nrow)
                    except Exception as e:
                        logger.warning(f"Não foi possível extrair métricas do leaderboard: {e}")
                        mlflow.log_metric("total_models_trained", 0)
                
                mlflow.log_metric("best_model_score", best_model_score)
                mlflow.log_metric("training_duration", training_duration)
                
            except Exception as e:
                logger.warning(f"Erro ao processar métricas do leaderboard: {e}")
                # Valores padrão
                mlflow.log_metric("best_model_score", 0.0)
                mlflow.log_metric("training_duration", training_duration)
                mlflow.log_metric("total_models_trained", 0)
            
            # Tentar salvar leaderboard com tratamento de erro
            try:
                leaderboard_df = leaderboard.as_data_frame()
                leaderboard_path = f"h2o_leaderboard_{run_name}.csv"
                leaderboard_df.to_csv(leaderboard_path, index=False)
                mlflow.log_artifact(leaderboard_path)
            except Exception as e:
                logger.warning(f"Não foi possível salvar leaderboard como CSV: {e}")
                # Salvar como texto simples se CSV falhar
                try:
                    leaderboard_text = str(leaderboard.head(10))
                    leaderboard_path = f"h2o_leaderboard_{run_name}.txt"
                    with open(leaderboard_path, "w") as f:
                        f.write(f"H2O AutoML Leaderboard - {run_name}\n")
                        f.write("=" * 50 + "\n")
                        f.write(leaderboard_text)
                    mlflow.log_artifact(leaderboard_path)
                except Exception as e2:
                    logger.warning(f"Não foi possível salvar leaderboard como texto: {e2}")
            
            # Salvar modelo localmente (apenas se houver modelos)
            if hasattr(aml, 'leader') and aml.leader is not None:
                model_dir = "models/h2o_models"
                os.makedirs(model_dir, exist_ok=True)
                model_path = f"{model_dir}/h2o_model_{run_name}"
                
                # Salvar o melhor modelo (leader) em vez do AutoML object
                best_model = aml.leader
                h2o.save_model(best_model, path=model_path)
                logger.info(f"Modelo salvo em: {model_path}")
                
                # Logar modelo no MLflow
                temp_model_path = f"temp_h2o_model_{run_name}"
                os.makedirs(temp_model_path, exist_ok=True)
                h2o.save_model(best_model, path=temp_model_path)
                mlflow.log_artifacts(temp_model_path, artifact_path="model")
                
                # Limpar pasta temporária
                import shutil
                if os.path.exists(temp_model_path):
                    shutil.rmtree(temp_model_path)
            else:
                logger.warning("⚠️ Nenhum modelo para salvar (nenhum modelo foi treinado)")
                
                # Criar um arquivo placeholder explicando a situação
                no_model_path = f"no_model_{run_name}.txt"
                with open(no_model_path, "w") as f:
                    f.write(f"H2O AutoML - {run_name}\n")
                    f.write("=" * 50 + "\n")
                    f.write("Nenhum modelo foi treinado durante esta execução.\n")
                    f.write("Possíveis causas:\n")
                    f.write("1. Tempo de treinamento insuficiente\n")
                    f.write("2. Dados inadequados para os algoritmos\n")
                    f.write("3. Problemas de qualidade dos dados\n")
                    f.write(f"Tempo de treinamento: {training_duration:.2f} segundos\n")
                
                mlflow.log_artifact(no_model_path)
            
            # Gerar relatório de classificação para problemas de classificação (apenas se houver modelos)
            if (clean_data[target].dtype == 'object' or clean_data[target].nunique() < 20) and hasattr(aml, 'leader') and aml.leader is not None:
                try:
                    best_model = aml.leader
                    predictions = best_model.predict(h2o_frame)
                    pred_array = predictions['predict'].as_data_frame()['predict'].values
                    true_labels = clean_data[target].values
                    
                    # Calcular métricas
                    accuracy = accuracy_score(true_labels, pred_array)
                    f1_macro = f1_score(true_labels, pred_array, average='macro')
                    f1_weighted = f1_score(true_labels, pred_array, average='weighted')
                    
                    logger.info(f"\nMétricas de validação:")
                    logger.info(f"Accuracy: {accuracy:.4f}")
                    logger.info(f"F1-Score (macro): {f1_macro:.4f}")
                    logger.info(f"F1-Score (weighted): {f1_weighted:.4f}")
                    
                    # Log de métricas de validação
                    mlflow.log_metric("validation_accuracy", accuracy)
                    mlflow.log_metric("validation_f1_macro", f1_macro)
                    mlflow.log_metric("validation_f1_weighted", f1_weighted)
                    
                    # Gerar relatório
                    class_report = classification_report(true_labels, pred_array)
                    report_path = f"classification_report_{run_name}.txt"
                    with open(report_path, "w") as f:
                        f.write(f"Classification Report - H2O AutoML\n")
                        f.write(f"{'='*50}\n\n")
                        f.write(class_report)
                    
                    mlflow.log_artifact(report_path)
                    
                except Exception as e:
                    logger.warning(f"Não foi possível gerar relatório de classificação: {e}")
            else:
                logger.info("Pulando geração de relatório (não há modelos treinados ou não é problema de classificação)")
            
            # Limpar arquivos temporários
            if os.path.exists(leaderboard_path):
                os.remove(leaderboard_path)
            
            report_path_temp = f"classification_report_{run_name}.txt"
            if os.path.exists(report_path_temp):
                os.remove(report_path_temp)
            
            return aml, run.info.run_id
            
    except Exception as e:
        logger.error(f"Erro durante treinamento H2O: {e}")
        raise
    finally:
        cleanup_h2o()

def load_h2o_model(run_id: str):
    """
    Carrega modelo H2O do MLflow
    """
    import h2o
    
    # Inicializar H2O se não estiver ativo
    try:
        h2o.init(max_mem_size="2G", nthreads=-1)
    except:
        pass  # H2O já pode estar inicializado
    
    try:
        # Download do artefato
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        
        # Encontrar e carregar o modelo
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(".zip"):
                    model_path = os.path.join(root, file)
                    logger.info(f"Carregando modelo H2O de: {model_path}")
                    model = h2o.load_model(model_path)
                    
                    # Verificar se o modelo foi carregado corretamente
                    if model is None:
                        raise ValueError("Modelo carregado é None")
                    
                    logger.info(f"Modelo H2O carregado com sucesso: {type(model)}")
                    return model
        
        raise FileNotFoundError("Modelo H2O não encontrado nos artefatos.")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo H2O: {e}")
        raise

def predict_with_h2o(model, data: pd.DataFrame):
    """
    Faz predições usando modelo H2O
    """
    import h2o
    
    # Verificar se o modelo é válido
    if model is None:
        raise ValueError("Modelo H2O é None. Verifique se o modelo foi carregado corretamente.")
    
    try:
        logger.info(f"Iniciando predição com modelo H2O: {type(model)}")
        
        # Preparar dados da mesma forma que no treinamento
        h2o_frame, _ = prepare_data_for_h2o(data, target="dummy")  # target não usado para predição
        
        # Fazer predições
        predictions = model.predict(h2o_frame)
        pred_array = predictions['predict'].as_data_frame()['predict'].values
        
        logger.info(f"Predição concluída: {len(pred_array)} previsões")
        return pred_array
        
    except Exception as e:
        logger.error(f"Erro na predição H2O: {e}")
        raise
    finally:
        # Limpar frame H2O para liberar memória
        try:
            if 'h2o_frame' in locals():
                h2o_frame = None
        except:
            pass
