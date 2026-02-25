import streamlit as st
import pandas as pd
import os
import queue
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_data, get_data_summary
from src.autogluon_utils import train_model as train_autogluon, load_model_from_mlflow as load_autogluon
from src.flaml_utils import train_flaml_model, load_flaml_model
from src.log_utils import setup_logging_to_queue, StdoutRedirector
from src.mlflow_utils import heal_mlruns
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
        
        framework = st.selectbox("Selecione o Framework AutoML", ["AutoGluon", "FLAML"])
        target = st.selectbox("Selecione a coluna alvo (Target)", columns)
        run_name = st.text_input("Nome da Run", value=f"{framework.lower()}_run_{int(time.time())}")
        
        # Framework specific options
        st.subheader(f"Configurações para {framework}")
        if framework == "AutoGluon":
            time_limit = st.slider("Limite de tempo (segundos)", 30, 3600, 60)
            presets = st.selectbox("Presets", ['medium_quality', 'best_quality', 'high_quality', 'good_quality', 'optimize_for_deployment'])
        else: # FLAML
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
            
            for lib in ['flaml', 'autogluon', 'mlflow']:
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
                            res_predictor, res_run_id = train_autogluon(df, target, run_name, time_limit, presets)
                            result_queue.put({"predictor": res_predictor, "run_id": res_run_id, "type": "autogluon", "success": True})
                        else: # FLAML
                            res_automl, res_run_id = train_flaml_model(df, target, run_name, time_budget, task, metric, estimator_list)
                            result_queue.put({"predictor": res_automl, "run_id": res_run_id, "type": "flaml", "success": True})
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
                
                all_logs = []
                performance_history = []
                while not training_done.is_set():
                    new_logs = False
                    while not log_queue.empty():
                        log_line = log_queue.get()
                        all_logs.append(log_line)
                        new_logs = True
                    
                    if new_logs:
                        log_placeholder.code("\n".join(all_logs[-20:])) # Mostrar mais linhas
                        
                        # Parse performance metrics from logs
                        for log_line in all_logs[-5:]: # Verificar apenas as linhas mais recentes
                            # Padrões comuns de log do FLAML e AutoGluon
                            # Ex: "Iteração 5: loss=0.1234", "Best loss: 0.1234", "val_score: 0.85"
                            try:
                                import re
                                # Procurar por padrões numéricos após palavras-chave
                                if any(kw in log_line.lower() for kw in ["loss", "score", "accuracy", "val_"]):
                                    # Extrair o último número da linha que parece um float
                                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", log_line)
                                    if numbers:
                                        val = float(numbers[-1])
                                        # Evitar adicionar números que parecem timestamps ou IDs (valores muito grandes)
                                        if abs(val) < 1000000:
                                            performance_history.append(val)
                                            chart_placeholder.line_chart(performance_history)
                            except:
                                pass
                    
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
        m_type = col1.selectbox("Tipo do Modelo", ["AutoGluon", "FLAML"])
        run_id_input = col2.text_input("Run ID")
        
        if st.button("Carregar Modelo"):
            try:
                if m_type == "AutoGluon":
                    st.session_state['predictor'] = load_autogluon(run_id_input)
                    st.session_state['model_type'] = "autogluon"
                else:
                    st.session_state['predictor'] = load_flaml_model(run_id_input)
                    st.session_state['model_type'] = "flaml"
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
                    if m_type == "autogluon":
                        predictions = predictor.predict(predict_df)
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

    exp_name = st.selectbox("Selecione o Experimento", ["AutoGluon_Experiments", "FLAML_Experiments"])
    
    try:
        # Heal again if we are about to access the history
        heal_mlruns()
        # Check if experiment exists first to avoid crashes on malformed file store
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            st.info(f"O experimento '{exp_name}' ainda não foi criado. Inicie um treinamento primeiro.")
        else:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                st.dataframe(runs)
            else:
                st.write("Nenhuma run encontrada para este experimento.")
    except Exception as e:
        st.error(f"Erro ao acessar o MLflow: {e}")
        st.warning("Isso pode ser causado por arquivos de metadados corrompidos na pasta 'mlruns'. Use o botão 'Limpar Cache' na barra lateral se o erro persistir.")
