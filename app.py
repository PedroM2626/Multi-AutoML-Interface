import streamlit as st
import pandas as pd
import os
from src.data_utils import load_data, get_data_summary
from src.autogluon_utils import train_model, get_leaderboard
import mlflow

st.set_page_config(page_title="AutoGluon Streamlit UI", layout="wide")

st.title("🚀 AutoGluon Visual Interface")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'predictor' not in st.session_state:
    st.session_state['predictor'] = None
if 'run_id' not in st.session_state:
    st.session_state['run_id'] = None

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
        
        target = st.selectbox("Selecione a coluna alvo (Target)", columns)
        time_limit = st.slider("Limite de tempo (segundos)", 30, 3600, 60)
        presets = st.selectbox("Presets de Qualidade", ['best_quality', 'high_quality', 'good_quality', 'medium_quality', 'optimize_for_deployment', 'interpretable', 'ignore_text'])
        run_name = st.text_input("Nome da Run", value="autogluon_run")
        
        if st.button("Iniciar Treinamento"):
            with st.spinner("Treinando modelo com AutoGluon..."):
                try:
                    predictor, run_id = train_model(df, target, run_name, time_limit, presets)
                    st.session_state['predictor'] = predictor
                    st.session_state['run_id'] = run_id
                    st.success(f"Treinamento concluído! Run ID: {run_id}")
                    
                    st.subheader("Leaderboard")
                    leaderboard = get_leaderboard(predictor)
                    st.dataframe(leaderboard)
                    
                except Exception as e:
                    st.error(f"Erro durante o treinamento: {e}")
    else:
        st.warning("Por favor, faça o upload de dados primeiro.")

elif menu == "Predição":
    st.header("🔮 Predição")
    
    # Option to load existing model
    load_option = st.radio("Escolha o modelo", ["Modelo da sessão atual", "Carregar de uma Run ID do MLflow"])
    
    if load_option == "Carregar de uma Run ID do MLflow":
        run_id_input = st.text_input("Digite a Run ID do MLflow")
        if st.button("Carregar Modelo"):
            with st.spinner("Carregando modelo do MLflow..."):
                try:
                    from src.autogluon_utils import load_model_from_mlflow
                    predictor = load_model_from_mlflow(run_id_input)
                    st.session_state['predictor'] = predictor
                    st.session_state['run_id'] = run_id_input
                    st.success("Modelo carregado com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao carregar modelo: {e}")

    if st.session_state['predictor'] is not None:
        predictor = st.session_state['predictor']
        
        st.subheader("Upload de dados para predição")
        predict_file = st.file_uploader("Escolha o arquivo para predição", type=["csv", "xlsx", "xls"], key="predict_upload")
        
        if predict_file is not None:
            predict_df = load_data(predict_file)
            st.dataframe(predict_df.head())
            
            if st.button("Executar Predição"):
                try:
                    predictions = predictor.predict(predict_df)
                    result_df = predict_df.copy()
                    result_df['Predictions'] = predictions
                    
                    st.success("Predições concluídas!")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predições (CSV)",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Erro na predição: {e}")
    else:
        st.warning("Nenhum modelo treinado disponível. Treine um modelo primeiro ou carregue um de uma Run.")

elif menu == "Histórico (MLflow)":
    st.header("📊 Histórico de Experimentos (MLflow)")
    st.info("Para visualizar os experimentos detalhadamente, execute `mlflow ui` no terminal.")
    
    try:
        experiments = mlflow.search_runs(experiment_names=["AutoGluon_Experiments"])
        if not experiments.empty:
            st.dataframe(experiments)
        else:
            st.write("Nenhum experimento encontrado.")
    except Exception as e:
        st.error(f"Erro ao buscar experimentos: {e}")
