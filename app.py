import streamlit as st
import pandas as pd
import numpy as np
import os
import io
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import queue

# Forçar reload dos módulos para pegar as alterações mais recentes
modules_to_reload = [
    'src.autogluon_utils',
    'src.flaml_utils', 
    'src.h2o_utils',
    'src.mlflow_cache'
]

for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

from src.data_utils import load_data, get_data_summary
from src.autogluon_utils import train_model as train_autogluon, load_model_from_mlflow as load_autogluon
from src.flaml_utils import train_flaml_model, load_flaml_model
from src.h2o_utils import train_h2o_model, load_h2o_model
from src.log_utils import setup_logging_to_queue, StdoutRedirector
from src.mlflow_utils import heal_mlruns
from src.mlflow_cache import mlflow_cache, get_cached_experiment_list
import mlflow
import time

st.set_page_config(page_title="AutoML Visual Interface", layout="wide")

# Heal MLflow cache on startup
heal_mlruns()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = None
if 'log_queue' not in st.session_state:
    st.session_state['log_queue'] = queue.Queue()

st.title("🚀 AutoML Multi-Framework Interface")

# Sidebar navigation
menu = st.sidebar.selectbox("Menu", ["Upload de Dados", "Treinamento", "Predição", "Histórico (MLflow)"])

if menu == "Upload de Dados":
    st.header("📂 Upload de Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state['df'] = df
            st.success("Arquivo carregado com sucesso!")
            
            st.subheader("Visualização dos Dados")
            st.dataframe(df.head())
            
            st.subheader("Resumo dos Dados")
            summary = get_data_summary(df)
            col1, col2, col3 = st.columns(3)
            col1.metric("Linhas", summary['rows'])
            col2.metric("Colunas", summary['columns'])
            
            st.write("Tipos de Dados e Valores Ausentes:")
            summary_df = pd.DataFrame({
                "Tipo": summary['dtypes'],
                "Ausentes": summary['missing_values']
            })
            st.table(summary_df)
            
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo: {e}")

elif menu == "Treinamento":
    st.header("🧠 Treinamento de Modelo")
    
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        columns = df.columns.tolist()
        
        framework = st.selectbox("Selecione o Framework AutoML", ["AutoGluon", "FLAML", "H2O AutoML"])
        target = st.selectbox("Selecione a coluna alvo (Target)", columns)
        run_name = st.text_input("Nome da Run", value=f"{framework.lower()}_run_{int(time.time())}")
        
        # Framework specific options
        st.subheader(f"Configurações para {framework}")
        
        # Opções comuns para todos os frameworks
        seed = st.number_input("Seed (reprodutibilidade)", value=42, min_value=0, max_value=9999)
        
        # Inicializar variáveis para todos os frameworks
        time_limit = time_budget = max_runtime_secs = 60
        presets = task = metric = estimator_list = None
        nfolds = balance_classes = sort_metric = exclude_algos = None
        
        if framework == "AutoGluon":
            time_limit = st.slider("Limite de tempo (segundos)", 30, 3600, 60)
            presets = st.selectbox("Presets", ['medium_quality', 'best_quality', 'high_quality', 'good_quality', 'optimize_for_deployment'])
        elif framework == "FLAML":
            time_budget = st.slider("Budget de tempo (segundos)", 30, 3600, 60)
            task = st.selectbox("Tarefa", ['classification', 'regression', 'ts_forecast', 'rank'])
            
            # Smart metric selection for FLAML
            num_classes = df[target].nunique()
            if task == 'classification':
                if num_classes > 2:
                    st.warning(f"Detectado problema multiclasse ({num_classes} classes).")
                    metric_options = ['auto', 'accuracy', 'macro_f1', 'micro_f1', 'roc_auc_ovr', 'roc_auc_ovo', 'log_loss']
                else:
                    metric_options = ['auto', 'accuracy', 'roc_auc', 'f1', 'log_loss']
            elif task == 'regression':
                metric_options = ['auto', 'rmse', 'mae', 'r2', 'mape']
            else:
                metric_options = ['auto']
                
            metric = st.selectbox("Métrica", metric_options)
            estimators = st.multiselect("Estimadores", ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree', 'lrl1', 'lrl2'], default=['lgbm', 'rf'])
            estimator_list = estimators if estimators else 'auto'
        elif framework == "H2O AutoML":
            st.warning("⚠️ H2O AutoML requer Java instalado. Se não tiver Java, use AutoGluon ou FLAML.")
            st.info("💡 Para usar H2O sem instalar Java localmente, use Docker.")
            
            max_runtime_secs = st.slider("Tempo máximo (segundos)", 60, 3600, 300)
            max_models = st.slider("Número máximo de modelos", 5, 50, 10)
            nfolds = st.slider("Número de folds CV", 2, 10, 3)
            balance_classes = st.checkbox("Balancear classes", value=True)
            
            # Opções avançadas H2O
            with st.expander("⚙️ Opções Avançadas H2O"):
                sort_metric = st.selectbox("Métrica de ordenação", ["AUTO", "AUC", "logloss", "RMSE", "MAE", "F1"])
                
                exclude_options = ['DeepLearning', 'GLM', 'GBM', 'DRF', 'XGBoost', 'GLRM']
                exclude_algos = st.multiselect("Excluir algoritmos", exclude_options, help="Algoritmos para excluir do AutoML")

        if st.button("Iniciar Treinamento"):
            st.subheader("📺 Monitoramento em Tempo Real")
            
            col_logs, col_chart = st.columns([1, 1])
            
            with col_logs:
                st.write("📋 Logs de Treinamento")
                log_placeholder = st.empty()
            
            with col_chart:
                st.write("📈 Evolução da Performance")
                chart_placeholder = st.empty()
            
            # Shared state for thread communication
            import threading
            training_done = threading.Event()
            log_queue = queue.Queue()
            result_queue = queue.Queue()
            
            # Custom Log Handler for the thread
            class ThreadLogHandler(logging.Handler):
                def emit(self, record):
                    msg = self.format(record)
                    log_queue.put(msg)
            
            # Setup logger to capture all framework output
            logger = logging.getLogger()
            for h in logger.handlers[:]:
                logger.removeHandler(h)
                
            handler = ThreadLogHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            
            for lib in ['flaml', 'autogluon', 'mlflow', 'h2o']:
                l = logging.getLogger(lib)
                l.addHandler(handler)
                l.setLevel(logging.INFO)
            
            # Training function to run in thread
            def run_training():
                # Redirect stdout and stderr to capture everything
                import io
                from contextlib import redirect_stdout, redirect_stderr
                
                class LogIO(io.StringIO):
                    def write(self, s):
                        if s.strip():
                            log_queue.put(s.strip())
                        return super().write(s)

                with redirect_stdout(LogIO()), redirect_stderr(LogIO()):
                    try:
                        if framework == "AutoGluon":
                            res_predictor, res_run_id = train_autogluon(df, target, run_name, time_limit, presets, seed)
                            result_queue.put({"predictor": res_predictor, "run_id": res_run_id, "type": "autogluon", "success": True})
                        elif framework == "FLAML":
                            res_automl, res_run_id = train_flaml_model(df, target, run_name, time_budget, task, metric, estimator_list, seed)
                            result_queue.put({"predictor": res_automl, "run_id": res_run_id, "type": "flaml", "success": True})
                        elif framework == "H2O AutoML":
                            res_automl, res_run_id = train_h2o_model(
                                df, target, run_name, 
                                max_runtime_secs=max_runtime_secs, 
                                max_models=max_models, 
                                nfolds=nfolds, 
                                balance_classes=balance_classes, 
                                seed=seed, 
                                sort_metric=sort_metric, 
                                exclude_algos=exclude_algos
                            )
                            result_queue.put({"predictor": res_automl, "run_id": res_run_id, "type": "h2o", "success": True})
                    except Exception as e:
                        import traceback
                        error_msg = f"ERRO CRÍTICO NO TREINAMENTO: {str(e)}\n{traceback.format_exc()}"
                        log_queue.put(error_msg)
                        result_queue.put({"success": False, "error": str(e)})
                    finally:
                        training_done.set()
                        logger.removeHandler(handler)

            # Start training
            try:
                thread = threading.Thread(target=run_training)
                thread.start()
                
                # UI Update Loop
                all_logs = []
                performance_history = []
                last_pos = 0
                
                # Clear old flaml.log if exists
                if os.path.exists("flaml.log"):
                    try:
                        os.remove("flaml.log")
                    except:
                        pass

                while not training_done.is_set():
                    # 1. Capture logs from the queue (Standard logging/stdout)
                    new_logs = False
                    while not log_queue.empty():
                        log_line = log_queue.get()
                        # Filter out the annoying FLAML low-cost warning from UI logs
                        if "low-cost partial config" in log_line.lower():
                            continue
                            
                        if log_line not in all_logs:
                            all_logs.append(log_line)
                            new_logs = True
                    
                    # 2. Capture logs from flaml.log file (Iterative progress)
                    if os.path.exists("flaml.log"):
                        try:
                            with open("flaml.log", "r") as f:
                                f.seek(last_pos)
                                new_lines = f.readlines()
                                last_pos = f.tell()
                                if new_lines:
                                    for line in new_lines:
                                        clean_line = line.strip()
                                        # Filter warning from file logs too
                                        if "low-cost partial config" in clean_line.lower():
                                            continue
                                            
                                        if clean_line and clean_line not in all_logs:
                                            all_logs.append(clean_line)
                                            new_logs = True
                                            
                                            # 3. Parse performance metrics
                                            if any(kw in clean_line.lower() for kw in ["loss", "accuracy", "score"]):
                                                import re
                                                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_line)
                                                if numbers:
                                                    try:
                                                        val = float(numbers[-1])
                                                        if 0 <= abs(val) <= 1000:
                                                            performance_history.append(val)
                                                            chart_placeholder.line_chart(performance_history)
                                                    except:
                                                        pass
                        except:
                            pass

                    if new_logs:
                        # Display only the most relevant recent logs
                        log_placeholder.code("\n".join(all_logs[-20:]))
                    
                    time.sleep(0.5)
                
                # Get final results from queue
                final_result = result_queue.get()
                
                if final_result["success"]:
                    st.session_state['predictor'] = final_result["predictor"]
                    st.session_state['run_id'] = final_result["run_id"]
                    st.session_state['model_type'] = final_result["type"]
                    st.success(f"Treinamento finalizado com sucesso! Run ID: {final_result['run_id']}")
                else:
                    st.error(f"O treinamento falhou: {final_result['error']}")

                # Show all logs at the end
                while not log_queue.empty():
                    all_logs.append(log_queue.get())
                
                if all_logs:
                    with st.expander("Ver Logs de Treinamento Completos"):
                        st.code("\n".join(all_logs))
                
                # Post-training visualizations
                if final_result["success"]:
                    if final_result['type'] == "flaml":
                        predictor = final_result['predictor']
                        
                        st.subheader("🏆 Melhor Modelo (FLAML)")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Melhor Estimador", predictor.best_estimator)
                        col2.metric("Melhor Perda (Loss)", f"{predictor.best_loss:.4f}")
                        col3.metric("Melhor Iteração", predictor.best_iteration)
                        
                        with st.expander("⚙️ Melhor Configuração (Hiperparâmetros)"):
                            st.json(predictor.best_config)
                            
                        if hasattr(predictor, 'best_config_per_estimator') and predictor.best_config_per_estimator:
                            with st.expander("📊 Melhores Configurações por Estimador"):
                                st.json(predictor.best_config_per_estimator)
                            
                        if hasattr(predictor, 'feature_importances_') and predictor.feature_importances_ is not None:
                            try:
                                fig, ax = plt.subplots()
                                importances = predictor.feature_importances_
                                if hasattr(predictor, 'feature_names_in_'):
                                    feature_names = predictor.feature_names_in_
                                else:
                                    feature_names = df.drop(columns=[target]).columns
                                
                                if len(importances) == len(feature_names):
                                    feat_importances = pd.Series(importances, index=feature_names)
                                    feat_importances.nlargest(10).plot(kind='barh', ax=ax)
                                    plt.title("Top 10 Feature Importances (FLAML)")
                                    st.pyplot(fig)
                                else:
                                    st.info(f"Importância de variáveis disponível, mas com mismatch de colunas ({len(importances)} vs {len(feature_names)}).")
                            except Exception as feat_err:
                                st.warning(f"Erro ao gerar gráfico de importância: {feat_err}")
                    
                    elif final_result['type'] == "autogluon":
                        predictor = final_result['predictor']
                        st.subheader("🏆 Resultados do AutoGluon")
                        
                        best_model = predictor.get_model_best()
                        st.success(f"O melhor modelo encontrado foi: **{best_model}**")
                        
                        st.subheader("Leaderboard Final")
                        leaderboard = predictor.leaderboard(silent=True)
                        st.dataframe(leaderboard)
                        
                        with st.expander("⚙️ Detalhes de Treinamento (AutoGluon Info)"):
                            try:
                                info = predictor.info()
                                st.json(info)
                            except:
                                st.write("Informações detalhadas não disponíveis para este modelo.")
                        
                        if st.checkbox("Gerar Importância de Variáveis (AutoGluon)"):
                            with st.spinner("Calculando importância (isso pode levar um tempo)..."):
                                try:
                                    fi = predictor.feature_importance(df)
                                    st.dataframe(fi)
                                    fig, ax = plt.subplots()
                                    fi['importance'].nlargest(10).plot(kind='barh', ax=ax)
                                    plt.title("Feature Importance (AutoGluon)")
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Erro ao calcular importância: {e}")
                    
                    elif final_result['type'] == "h2o":
                        automl = final_result['predictor']
                        st.subheader("🏆 Resultados do H2O AutoML")
                        
                        # Verificar se o H2O ainda está conectado antes de acessar o modelo
                        try:
                            best_model = automl.leader
                            if best_model is not None:
                                st.success(f"O melhor modelo encontrado foi: **{best_model.model_id}**")
                                
                                st.subheader("Leaderboard Final")
                                try:
                                    leaderboard = automl.leaderboard.as_data_frame()
                                    st.dataframe(leaderboard)
                                except Exception as e:
                                    st.warning(f"Não foi possível exibir o leaderboard: {e}")
                                    # Tentar exibir como texto
                                    try:
                                        st.text(str(automl.leaderboard.head(10)))
                                    except:
                                        st.info("Leaderboard não disponível (conexão H2O encerrada)")
                                
                                with st.expander("⚙️ Detalhes do Melhor Modelo (H2O)"):
                                    try:
                                        model_params = {
                                            "model_id": best_model.model_id,
                                            "algo": best_model.algo,
                                            "model_type": best_model._model_json["output"]["model_category"]
                                        }
                                        st.json(model_params)
                                    except Exception as e:
                                        st.warning(f"Não foi possível obter detalhes do modelo: {e}")
                            else:
                                st.warning("⚠️ Nenhum modelo foi treinado durante esta execução.")
                                st.info("Isso pode acontecer quando:")
                                st.info("• O tempo máximo é insuficiente para o dataset")
                                st.info("• Os dados não são adequados para os algoritmos selecionados")
                                st.info("• Houver problemas na configuração dos parâmetros")
                                
                                # Tentar mostrar informações básicas
                                try:
                                    st.subheader("📊 Informações do Treinamento")
                                    st.info(f"• Tipo: H2O AutoML")
                                    st.info(f"• Run ID: {final_result['run_id']}")
                                    st.info(f"• Status: Concluído, mas sem modelos treinados")
                                    st.info(f"• Duração: ~3600 segundos (timeout)")
                                    st.info(f"• Recomendação: Aumentar tempo máximo ou reduzir complexidade dos dados")
                                except:
                                    pass
                        except Exception as e:
                            st.error(f"⚠️ Não foi possível acessar os detalhes do modelo H2O: {e}")
                            st.info("Isso acontece quando o H2O é finalizado após o treinamento. Os resultados foram salvos no MLflow com sucesso!")
                            
                            # Exibir informações básicas do AutoML
                            try:
                                st.info(f"📊 **Informações do Treinamento:**")
                                st.info(f"• Tipo: H2O AutoML")
                                st.info(f"• Run ID: {final_result['run_id']}")
                                st.info(f"• Status: Concluído com sucesso")
                                st.info(f"• Métricas registradas no MLflow")
                            except:
                                pass

            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                st.error(f"Erro durante o treinamento: {e}")
                with st.expander("Ver detalhes do erro (Traceback)"):
                    st.code(error_details)
            finally:
                pass
    else:
        st.warning("Por favor, faça o upload de dados primeiro.")

elif menu == "Predição":
    st.header("🔮 Predição")
    
    load_option = st.radio("Escolha o modelo", ["Modelo da sessão atual", "Carregar do MLflow"])
    
    if load_option == "Carregar do MLflow":
        col1, col2 = st.columns(2)
        m_type = col1.selectbox("Tipo do Modelo", ["AutoGluon", "FLAML", "H2O AutoML"])
        run_id_input = col2.text_input("Run ID")
        
        if st.button("Carregar Modelo"):
            try:
                if m_type == "AutoGluon":
                    st.session_state['predictor'] = load_autogluon(run_id_input)
                    st.session_state['model_type'] = "autogluon"
                elif m_type == "FLAML":
                    st.session_state['predictor'] = load_flaml_model(run_id_input)
                    st.session_state['model_type'] = "flaml"
                elif m_type == "H2O AutoML":
                    st.session_state['predictor'] = load_h2o_model(run_id_input)
                    st.session_state['model_type'] = "h2o"
                st.success("Modelo carregado com sucesso!")
            except Exception as e:
                st.error(f"Erro ao carregar: {e}")

    if st.session_state['predictor'] is not None:
        predictor = st.session_state['predictor']
        m_type = st.session_state['model_type']
        
        st.info(f"Modelo ativo: {m_type}")
        
        predict_file = st.file_uploader("Escolha o arquivo para predição", type=["csv", "xlsx", "xls"])
        
        if predict_file is not None:
            predict_df = load_data(predict_file)
            st.dataframe(predict_df.head())
            
            if st.button("Executar Predição"):
                try:
                    # Verificar se o predictor não é None
                    if predictor is None:
                        st.error("Nenhum modelo carregado. Por favor, carregue um modelo primeiro.")
                        st.stop()
                    
                    if m_type == "autogluon":
                        predictions = predictor.predict(predict_df)
                    elif m_type == "h2o":
                        from src.h2o_utils import predict_with_h2o
                        predictions = predict_with_h2o(predictor, predict_df)
                    else: # flaml
                        predictions = predictor.predict(predict_df)
                    
                    result_df = predict_df.copy()
                    result_df['Predictions'] = predictions
                    
                    st.success("Predições concluídas!")
                    st.dataframe(result_df)
                    
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Erro na predição: {e}")

elif menu == "Histórico (MLflow)":
    st.header("📊 Histórico de Experimentos")
    
    # Button to clean corrupted MLflow metadata
    if st.sidebar.button("Limpar Cache MLflow (Reparar Erros)"):
        import shutil
        if os.path.exists("mlruns"):
            # Instead of deleting everything, we could try to find the malformed ones
            # but deleting is safer for a local "repair"
            shutil.rmtree("mlruns")
            st.sidebar.success("Cache limpo! Reinicie o treinamento.")
            st.rerun()

    # Botão para limpar cache MLflow
    if st.sidebar.button("Limpar Cache MLflow"):
        mlflow_cache.clear_cache()
        st.sidebar.success("Cache limpo!")
        st.rerun()

    # Usar lista cacheada de experimentos
    experiment_list = get_cached_experiment_list()
    exp_name = st.selectbox("Selecione o Experimento", experiment_list)
    
    try:
        # Usar cache para obter runs
        runs = mlflow_cache.get_cached_all_runs(exp_name)
        
        if not runs.empty:
            st.dataframe(runs)
            
            # Mostrar estatísticas do cache
            with st.expander("📊 Estatísticas do Cache"):
                st.write(f"Experimento: {exp_name}")
                st.write(f"Total de runs: {len(runs)}")
                st.write(f"Cache TTL: 5 minutos")
        else:
            st.write("Nenhuma run encontrada para este experimento.")
    except Exception as e:
        st.error(f"Erro ao acessar o MLflow: {e}")
        st.warning("Isso pode ser causado por arquivos de metadados corrompidos na pasta 'mlruns'. Use o botão 'Limpar Cache' na barra lateral se o erro persistir.")
