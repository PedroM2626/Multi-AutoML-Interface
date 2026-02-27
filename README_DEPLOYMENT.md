# 🚀 Deployment Guide - Multi-AutoML Interface

## 📋 Overview

Este guia cobre as opções de deployment do Multi-AutoML Interface para ambientes remotos, incluindo configuração SSL, Docker, e plataformas cloud.

---

## 🌐 Opções de Deployment

### 1. GitHub Pages (Recomendado para Portfólio)

#### **✅ Vantagens:**
- Gratuito e ilimitado
- Integrado com GitHub
- SSL automático
- Fácil atualização
- Ideal para portfólio

#### **🔧 Configuração:**
```bash
# 1. Build para produção
npm run build

# 2. Deploy para GitHub Pages
npm run deploy

# 3. Acessar em minutos
https://seu-usuario.github.io/Multi-AutoML-Interface/
```

---

### 2. Heroku (Ideal para Protótipos)

#### **✅ Vantagens:**
- Free tier disponível
- Deploy automático com GitHub
- SSL automático
- Add-ons disponíveis
- Escalável

#### **🔧 Configuração:**
```bash
# 1. Instalar Heroku CLI
npm install -g heroku

# 2. Login no Heroku
heroku login

# 3. Criar app
heroku create seu-app-name

# 4. Configurar variáveis
heroku config:set MLFLOW_TRACKING_URI=database-url

# 5. Deploy
git push heroku main
```

#### **📝 Procfile:**
```dockerfile
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

### 3. Railway (Moderno e Simples)

#### **✅ Vantagens:**
- Interface amigável
- GitHub integration
- SSL automático
- Bom free tier
- Deploy rápido

#### **🔧 Configuração:**
```bash
# 1. Instalar Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Criar projeto
railway init

# 4. Deploy
railway up
```

---

### 4. Verc (Frontend Focado)

#### **✅ Vantagens:**
- Excelente para frontend
- Deploy instantâneo
- SSL automático
- Global CDN
- Preview automático

#### **🔧 Configuração:**
```bash
# 1. Instalar Vercel
npm install -g vercel

# 2. Deploy
vercel --prod

# 3. Configurar domínio (opcional)
vercel domains add seu-dominio.com
```

---

### 5. AWS EC2 (Produção Empresarial)

#### **✅ Vantagens:**
- Controle total
- Alta performance
- Segurança avançada
- Escalabilidade infinita
- Integração AWS completa

#### **🔧 Configuração:**
```bash
# 1. Criar EC2 instance
aws ec2 run-instances --image-id ami-12345 --instance-type t3.medium

# 2. Conectar via SSH
ssh -i key.pem ec2-user@ip-da-instancia

# 3. Instalar dependências
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt

# 4. Configurar ambiente
export MLFLOW_TRACKING_URI=http://localhost:5000
export STREAMLIT_SERVER_PORT=8501

# 5. Rodar aplicação
streamlit run app.py --server.port 8501
```

---

### 6. Google Cloud Platform

#### **✅ Vantagens:**
- Crédito gratuito generoso
- Integração com GitHub
- SSL automático
- Global CDN
- Monitoramento avançado

#### **🔧 Configuração:**
```bash
# 1. Instalar gcloud CLI
curl https://sdk.cloud.google.com | bash

# 2. Login
gcloud auth login

# 3. Criar projeto
gcloud projects create seu-projeto

# 4. Deploy
gcloud app deploy
```

---

## 🔒 Configuração SSL

### Opção A: SSL Automático (Recomendado)

#### **GitHub Pages:**
```bash
# SSL automático e gratuito
https://seu-usuario.github.io/Multi-AutoML-Interface/
```

#### **Heroku:**
```bash
# SSL automático em *.herokuapp.com
https://seu-app.herokuapp.com
```

#### **Vercel:**
```bash
# SSL automático em *.vercel.app
https://seu-app.vercel.app
```

### Opção B: Certificado Próprio

#### **Cloudflare (SSL Gratuito):**
```bash
# 1. Usar Cloudflare para SSL
# 2. Apontar domínio para app
# 3. Configurar SSL flexível no Cloudflare
```

#### **Let's Encrypt (SSL Gratuito):**
```bash
# 1. Instalar Certbot
sudo apt install certbot

# 2. Gerar certificado
sudo certbot certonly --standalone -d seu-dominio.com

# 3. Configurar no servidor
# Adicionar certificado à configuração do servidor
```

---

## 🐳 Docker para Produção

### Dockerfile Otimizado:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar aplicação
COPY . .

# Criar usuário non-root
RUN useradd --create-home --shell /bin/bash app
USER app

# Expor porta
EXPOSE 8501

# Variáveis de ambiente
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Comando de execução
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
```

### Docker Compose:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - MLFLOW_TRACKING_URI=/app/mlruns
      - STREAMLIT_SERVER_PORT=8501
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
    restart: unless-stopped

  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=/app/mlruns
    volumes:
      - ./mlruns:/app/mlruns
    command: mlflow server --host 0.0.0.0 --port 5000
    restart: unless-stopped
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions:
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v3
      
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        
    - name: Run tests
      run: |
        python -m pytest tests/
        
    - name: Deploy to Vercel
      uses: amondnet/vercel-action@v20
      with:
        vercel-token: ${{ secrets.VERCEL_TOKEN }}
        vercel-org-id: ${{ secrets.VERCEL_ORG_ID }}
        vercel-project-id: ${{ secrets.VERCEL_PROJECT_ID }}
```

---

## 🌐 Domínio Personalizado

### Configuração de DNS:
```bash
# Exemplo para domínio próprio
# 1. Apontar A record para IP do servidor
seu-dominio.com A 192.168.1.100

# 2. Configurar CNAME (se usar plataforma)
www.seu-dominio.com CNAME seu-app.vercel.app

# 3. Configurar MX records (email)
seu-dominio.com MX 10 mail.seu-provedor.com
```

---

## 📊 Monitoramento e Logging

### Configuração Produção:
```python
# logging_config.py
import logging
import os

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.environ.get('LOG_FILE', 'app.log'),
            'maxBytes': 10485760,  # 100MB
            'backupCount': 5,
            'formatter': 'standard',
        },
    },
    'loggers': {
        '': {
            'handlers': ['file'],
            'level': os.environ.get('LOG_LEVEL', 'INFO'),
        },
    }
}
```

---

## 🔐 Segurança em Produção

### Boas Práticas:
```bash
# 1. Variáveis de ambiente
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export MLFLOW_TRACKING_URI=/app/mlruns

# 2. Firewall
ufw allow 8501/tcp
ufw allow 5000/tcp

# 3. HTTPS com Nginx
server {
    listen 443 ssl;
    server_name seu-dominio.com;
    
    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📈 Performance e Escalabilidade

### Otimizações:
```python
# config.py
import streamlit as st

# Configurações de produção
st.set_page_config(
    page_title="Multi-AutoML Interface",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Cache de sessão
@st.cache_data(ttl=3600)  # 1 hora
def load_data():
    return expensive_operation()

# Limitar uso de memória
st.set_option('server.maxUploadSize', 200)  # 200MB
st.set_option('server.maxMessageSize', 1000)  # 1GB
```

---

## 🚀 Deploy Rápido

### Script Automático:
```bash
#!/bin/bash
# deploy.sh

echo "🚀 Iniciando deploy do Multi-AutoML Interface..."

# 1. Backup
git add .
git commit -m "Backup before deploy"

# 2. Testes
python -m pytest tests/ -v

# 3. Deploy para produção
vercel --prod

# 4. Notificação
curl -X POST https://hooks.slack.com/... \
  -H 'Content-type: application/json' \
  -d '{"text":"✅ Multi-AutoML Interface deployed!"}'

echo "🎉 Deploy concluído!"
```

---

## 📞 Suporte e Troubleshooting

### Problemas Comuns:
```bash
# 1. Erro de porta
netstat -tulpn | grep :8501

# 2. Logs de erro
docker logs container-name

# 3. Verificar variáveis
env | grep MLFLOW

# 4. Testar localmente
curl -I http://localhost:8501
```

### Contato Suporte:
- 📧 Issues no GitHub: https://github.com/PedroM2626/Multi-AutoML-Interface/issues
- 📧 Email: pedro.lahoz@email.com
- 📧 Documentação: README.md

---

## 🎯 Recomendações

### Para Portfólio:
✅ **GitHub Pages** - Melhor custo-benefício

### Para Produção:
✅ **Vercel** - Mais moderno e fácil
✅ **Heroku** - Bom para protótipos
✅ **AWS EC2** - Máximo controle e performance

### Para Empresas:
✅ **Google Cloud** - Integração completa
✅ **Azure** - Ecossistema Microsoft
✅ **DigitalOcean** - Bom custo-benefício

---

**Escolha a opção que melhor se adapta às suas necessidades!** 🚀
