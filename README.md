# 🚀 Multi-AutoML Interface

**Uma interface unificada para experimentação com AutoML, permitindo comparar múltiplos frameworks (AutoGluon, FLAML, H2O) com MLOps integrado via MLflow.**

---

## 🎯 **Visão Geral**

O Multi-AutoML Interface é uma aplicação web/desktop que simplifica o uso de frameworks AutoML, permitindo:

- **Comparação lado a lado** de diferentes engines AutoML
- **MLOps integrado** com tracking completo via MLflow
- **Interface unificada** para treinamento, avaliação e predição
- **Deploy flexível** (web, Docker, desktop)
- **Métricas e logging** detalhados

---

## ✨ **Features Principais**

### 🤖 **Frameworks AutoML Suportados:**
- **AutoGluon** (Amazon) - Performance excepcional
- **FLAML** (Microsoft) - Veloz e eficiente
- **H2O AutoML** (Enterprise) - Robusto e completo
- **TPOT** (Open Source) - Pipelines gerados por Algoritmos Genéticos

### 📊 **MLOps Integrado:**
- **MLflow tracking** completo
- **Data Lake versioning** automático com DVC
- **Experiment logging** automático
- **Model registry** centralizado
- **Performance metrics** detalhadas
- **Artifact management**

### 🖥️ **Multi-Deploy:**
- **Web interface** (Streamlit)
- **Docker container** (produção)
- **Desktop app** (Electron)
- **Hugging Face Spaces** (Live Demo)
- **Local development**

### 🎛️ **Interface Avançada:**
- **Upload de múltiplos datasets** (Treino, Validação, Teste)
- **Configuração avançada** de parâmetros
- **Monitoramento em tempo real**
- **Visualização de resultados**
- **Predição interativa**

---

## 🏗️ **Arquitetura**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │   ML Engines    │
│                 │    │                  │    │                 │
│ • Streamlit     │◄──►│ • Python         │◄──►│ • AutoGluon     │
│ • Electron      │    │ • FastAPI        │    │ • FLAML         │
│ • React         │    │ • MLflow         │    │ • H2O AutoML    │
│ • Custom UI     │    │ • Logging        │    │ • TPOT          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Storage       │    │   Monitoring     │    │   Deployment    │
│                 │    │                  │    │                 │
│ • File System   │    │ • MLflow UI      │    │ • Docker Hub    │
│ • MLflow Artifacts│  │ • Logs           │    │ • GitHub        │
│ • Model Registry│    │ • Metrics        │    │ • Electron Store│
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

---

## 🚀 **Quick Start**

### 📋 **Pré-requisitos:**
- **Python 3.11+**
- **Node.js 16+** (para desktop app)
- **Java 11+** (para H2O AutoML)
- **Git**

### 🔧 **Instalação:**

#### **1. Clonar Repositório:**
```bash
git clone https://github.com/PedroM2626/Multi-AutoML-Interface.git
cd Multi-AutoML-Interface
```

#### **2. Ambiente Python:**
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar (Windows)
venv\Scripts\activate

# Ativar (Mac/Linux)
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

#### **3. Iniciar MLflow:**
```bash
# Iniciar MLflow server
mlflow server --host 0.0.0.0 --port 5000
```

#### **4. Rodar Aplicação:**
```bash
# Opção 1: Web interface
streamlit run app.py --server.port 8501

# Opção 2: Desktop app (requer Node.js)
npm install && npm run dev

# Opção 3: Docker
docker-compose up
```

---

## 📖 **Guia de Uso**

### 🎯 **Workflow Básico:**

#### **1. Upload de Dados:**
- Formatos suportados: CSV, Excel
- **Múltiplos splits suportados**: Treino (obrigatório), Validação (opcional) e Teste (opcional)
- Detecção automática de tipos
- **Data Lake Automático**: Ao processar os dados, são copiados para a pasta `data_lake/` e versionados via DVC, com Hashes gerados para controle de versionamento.

#### **2. Configuração do Experimento:**
- **Framework**: AutoGluon, FLAML, H2O, TPOT
- **Target variable**: Coluna alvo
- **Parâmetros avançados**: seed, tempo, folds, max features textuais (TF-IDF), CV, etc.

#### **3. Treinamento:**
- **Monitoramento em tempo real**
- **Logs detalhados**
- **Progress tracking**

#### **4. Análise de Resultados:**
- **Leaderboards** comparativos
- **Performance metrics**
- **Model insights**

#### **5. Predição:**
- **Upload de novos dados**
- **Batch prediction**
- **Real-time inference**

---

## 🛠️ **Configuração Avançada**

### ⚙️ **Parâmetros dos Frameworks:**

#### **AutoGluon:**
```python
{
    'presets': 'best_quality',
    'time_limit': 3600,
    'seed': 42,
    'num_bag_folds': 5,
    'num_bag_sets': 1
}
```

#### **FLAML:**
```python
{
    'time_budget': 3600,
    'seed': 42,
    'ensemble': True,
    'metric': 'accuracy',
    'estimator_list': ['lgbm', 'xgboost', 'rf']
}
```

#### **H2O AutoML:**
```python
{
    'max_runtime_secs': 3600,
    'max_models': 20,
    'seed': 42,
    'nfolds': 5,
    'balance_classes': True,
    'sort_metric': 'AUTO'
}
```

#### **TPOT:**
```python
{
    'generations': 5,
    'population_size': 20,
    'cv': 5,
    'max_time_mins': 30,
    'config_dict': 'TPOT sparse',
    'tfidf_max_features': 500,
    'tfidf_ngram_range': (1, 2)
}
```

### 🎛️ **Configuração MLflow:**
```python
# Experiments
mlflow.set_experiment("AutoGluon_Experiments")
mlflow.set_experiment("FLAML_Experiments") 
mlflow.set_experiment("H2O_Experiments")

# Tracking
mlflow.log_param("framework", "autogluon")
mlflow.log_metric("accuracy", 0.95)
mlflow.log_artifact("model.pkl")
```

---

## 🐳 **Deploy com Docker**

### 📦 **Build e Execução:**

#### **1. Build da Imagem:**
```bash
docker build -t multi-automl:latest .
```

#### **2. Docker Compose:**
```bash
# Iniciar todos os serviços
docker-compose up -d

# Logs
docker-compose logs -f

# Parar
docker-compose down
```

#### **3. Portas:**
- **8501**: Streamlit UI
- **5000**: MLflow UI
- **54321**: H2O Cluster

---

## 🖥️ **Desktop App (Electron)**

### 📦 **Instalação e Build:**

#### **1. Instalar Node.js:**
```bash
# Download: https://nodejs.org/
node --version
npm --version
```

#### **2. Instalar Dependências:**
```bash
npm install
```

#### **3. Modo Desenvolvimento:**
```bash
npm run dev
```

#### **4. Build para Produção:**
```bash
# Windows
npm run build-win

# Mac
npm run build-mac

# Linux
npm run build-linux
```

#### **5. Features Desktop:**
- **Janela nativa** (sem navegador)
- **Menu profissional** com atalhos
- **File dialogs** nativos
- **System integration**
- **Offline mode**

---

## 📊 **Performance e Benchmarks**

### 🏆 **Comparação de Frameworks:**

| Framework | Velocidade | Performance | Memória | Facilidade |
|-----------|------------|--------------|---------|------------|
| **AutoGluon** | ⚡⚡⚡ | 🏆🏆 | 🏆🏆 | 🏆🏆🏆 |
| **FLAML** | ⚡⚡⚡⚡ | 🏆🏆 | 🏆🏆🏆 | 🏆🏆 |
| **H2O** | ⚡⚡ | 🏆🏆🏆 | 🏆 | 🏆 |
| **TPOT** | ⚡ | 🏆🏆🏆 | 🏆🏆 | 🏆 |

### 📈 **Métricas de Performance:**

#### **Dataset Teste (10k linhas, 50 colunas):**
```
AutoGluon: 2.5 min, 94.2% accuracy
FLAML: 1.8 min, 93.8% accuracy  
H2O: 4.2 min, 94.0% accuracy
```

#### **Uso de Memória:**
```
AutoGluon: ~2GB RAM
FLAML: ~1.5GB RAM
H2O: ~3GB RAM
TPOT: ~1GB RAM (Otimizado)
```

---

## 🔧 **Troubleshooting**

### ❌ **Problemas Comuns:**

#### **"Java não encontrado" (H2O):**
```bash
# Windows: Adicionar JAVA_HOME
set JAVA_HOME="C:\Program Files\Java\jdk-11"

# Mac/Linux: Exportar variável
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk
```

#### **"Porta já em uso":**
```bash
# Verificar portas
netstat -an | findstr 8501

# Matar processo
taskkill /PID <PID> /F

# Usar outra porta
streamlit run app.py --server.port 8502
```

#### **"Memory error":**
```bash
# Aumentar memória H2O
export H2O_MAX_MEM_SIZE="8G"

# Ou reduzir dataset
```

#### **"MLflow connection error" / "Missing mlruns":**
```bash
# Na nova versão, o diretório mlruns/.trash é cicatrizado e recriado automaticamente caso seja rompido.
# Para outros problemas:
mlflow server --host 0.0.0.0 --port 5000
```

---

## 🧪 **Testes**

### 📋 **Suite de Testes:**

#### **1. Testes de Integração:**
```bash
# Testar H2O integration
python tests/test_h2o_integration.py

# Testar MLflow integration  
python tests/test_mlflow_integration.py
```

#### **2. Testes Unitários:**
```bash
# Testar utils
pytest tests/test_utils.py

# Testar interface
pytest tests/test_interface.py
```

#### **3. Testes de Performance:**
```bash
# Benchmark frameworks
python tests/benchmark_frameworks.py
```

---

## 📁 **Estrutura do Projeto**

```
Multi-AutoML-Interface/
├── 📁 src/                    # Código fonte principal
│   ├── 📄 autogluon_utils.py  # AutoGluon integration
│   ├── 📄 flaml_utils.py      # FLAML integration
│   ├── 📄 h2o_utils.py        # H2O integration
│   ├── 📄 tpot_utils.py       # TPOT integration 
│   ├── 📄 mlflow_utils.py     # MLflow helpers e auto-healing
│   ├── 📄 mlflow_cache.py     # Cache otimizado
│   ├── 📄 data_utils.py       # Data processing
│   └── 📄 log_utils.py        # Logging utilities
├── 📁 tests/                  # Testes automatizados
│   ├── 📄 test_h2o_integration.py
│   ├── 📄 test_mlflow_integration.py
│   └── 📄 test_performance.py
├── 📁 electron/               # Desktop app (Electron)
│   ├── 📄 main.js             # Main process
│   ├── 📄 preload.js          # Security bridge
│   ├── 📄 renderer.js         # UI enhancements
│   └── 📁 assets/             # Icons e recursos
├── 📄 app.py                  # Streamlit main app
├── 📄 requirements.txt        # Python dependencies
├── 📄 package.json            # Node.js dependencies
├── 🐳 Dockerfile              # Docker configuration
├── 🐳 docker-compose.yml      # Multi-service setup
└── 📄 README.md               # Este arquivo
```

---

## 🤝 **Contribuição**

### 🎯 **Como Contribuir:**

#### **1. Fork e Clone:**
```bash
git clone https://github.com/PedroM2626/Multi-AutoML-Interface.git
cd Multi-AutoML-Interface
```

#### **2. Criar Branch:**
```bash
git checkout -b feature/nova-feature
```

#### **3. Desenvolver:**
- Seguir código style existente
- Adicionar testes
- Documentar mudanças

#### **4. Commit e Push:**
```bash
git add .
git commit -m "feat: add nova feature"
git push origin feature/nova-feature
```

#### **5. Pull Request:**
- Descrever mudanças
- Linkar issues
- Aguardar review

### 📝 **Guidelines:**
- **Python**: PEP 8
- **JavaScript**: ESLint
- **Commits**: Conventional Commits
- **Docs**: Markdown claro

---

## 📄 **Licença**

Este projeto está licenciado sob a **MIT License** - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🙏 **Créditos e Agradecimentos**

### 🤖 **Frameworks:**
- **AutoGluon** - Amazon Web Services
- **FLAML** - Microsoft Research  
- **H2O AutoML** - H2O.ai
- **TPOT** - Rhodes Lab
- **MLflow** - Databricks

### 🛠️ **Tecnologias:**
- **Streamlit** - Interface web
- **Electron** - Desktop app
- **Docker** - Containerização
- **FastAPI** - Backend API

### 📚 **Recursos:**
- **AutoML Documentation**
- **MLflow Tracking**
- **Streamlit Components**
- **Electron Security**

---

## 🗺️ **Roadmap Futuro**

### 🚀 **Próximas Features**
- [ ] **Auto-sklearn** (meta-learning)
- [ ] **Model explainability** (SHAP, LIME)
- [ ] **Advanced visualizations**
- [ ] **Batch processing**

---

### 🌐 **Live Demo:**
[Hugging Face Spaces - Multi-AutoML Interface](https://huggingface.co/spaces/PedroM2626/Multi-AutoML-Interface)

---

## 🎉 **Conclusão**

O **Multi-AutoML Interface** representa uma solução completa e profissional para experimentação com AutoML, combinando:

- **🤖 Múltiplos frameworks** em uma interface unificada
- **📊 MLOps integrado** com tracking completo
- **🖥️ Deploy flexível** (web, desktop, container)
- **🎛️ Interface intuitiva** para usuários técnicos
- **🔧 Configuração avançada** para experts
- **📈 Performance otimizada** para produção

**Ideal para:**
- **Data Scientists** que querem comparar frameworks
- **Pesquisadores** que experimentam diferentes abordagens
- **Estudantes** que aprendem sobre AutoML

---

*Desenvolvido por Pedro Morato Lahoz*