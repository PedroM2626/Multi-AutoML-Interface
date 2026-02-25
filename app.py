import streamlit as st
import pandas as pd
import os
import queue
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_utils import load_data, get_data_summary
from src.autogluon_utils import train_model as train_autogluon, load_model_from_mlflow as load_autogluon
from src.flaml_utils import train_flaml_model, load_flaml_model
from src.log_utils import setup_logging_to_queue, StdoutRedirector
import mlflow
import time

st.set_page_config(page_title="AutoML Visual Interface", layout="wide")

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
            metric = st.selectbox("Métrica", ['auto', 'accuracy', 'roc_auc', 'f1', 'rmse', 'mae', 'r2'])
            estimators = st.multiselect("Estimadores", ['lgbm', 'rf', 'catboost', 'xgboost', 'extra_tree', 'lrl1', 'lrl2'], default=['lgbm', 'rf'])
            estimator_list = estimators if estimators else 'auto'

        if st.button("Iniciar Treinamento"):
            log_container = st.empty()
            plot_container = st.empty()
            
            # Setup logging redirection
            old_stdout = sys.stdout
            sys.stdout = StdoutRedirector(st.session_state['log_queue'])
            
            logs = []
            
            try:
                with st.spinner(f"Treinando com {framework}..."):
                    if framework == "AutoGluon":
                        predictor, run_id = train_autogluon(df, target, run_name, time_limit, presets)
                        st.session_state['predictor'] = predictor
                        st.session_state['model_type'] = "autogluon"
                        
                        st.success(f"AutoGluon finalizado! Run ID: {run_id}")
                        st.subheader("Leaderboard")
                        st.dataframe(predictor.leaderboard(silent=True))
                    
                    else: # FLAML
                        automl, run_id = train_flaml_model(df, target, run_name, time_budget, task, metric, estimator_list)
                        st.session_state['predictor'] = automl
                        st.session_state['model_type'] = "flaml"
                        
                        st.success(f"FLAML finalizado! Run ID: {run_id}")
                        st.write(f"Melhor Estimador: {automl.best_estimator}")
                        st.write(f"Melhor Loss: {automl.best_loss}")
                        
                        # Show feature importance plot for FLAML
                        if hasattr(automl, 'feature_importances_'):
                            fig, ax = plt.subplots()
                            feat_importances = pd.Series(automl.feature_importances_, index=df.drop(columns=[target]).columns)
                            feat_importances.nlargest(10).plot(kind='barh', ax=ax)
                            plt.title("Top 10 Feature Importances (FLAML)")
                            st.pyplot(fig)

                # Capture logs while training (simplified for this context)
                while not st.session_state['log_queue'].empty():
                    logs.append(st.session_state['log_queue'].get())
                
                if logs:
                    with st.expander("Ver Logs de Treinamento"):
                        st.code("\n".join(logs))

            except Exception as e:
                st.error(f"Erro durante o treinamento: {e}")
            finally:
                sys.stdout = old_stdout
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
    exp_name = st.selectbox("Selecione o Experimento", ["AutoGluon_Experiments", "FLAML_Experiments"])
    
    try:
        runs = mlflow.search_runs(experiment_names=[exp_name])
        if not runs.empty:
            st.dataframe(runs)
        else:
            st.write("Nenhuma run encontrada.")
    except Exception as e:
        st.error(f"Erro ao buscar histórico: {e}")
