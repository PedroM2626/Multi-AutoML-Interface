# 🚀 Windows Deployment Guide - Multi-AutoML Interface

## 📋 Overview

Este guia cobre o deployment do Multi-AutoML Interface em ambiente Windows, incluindo Docker Desktop, configuração SSL e troubleshooting.

---

## 🔧 Pré-requisitos

### 1. Docker Desktop
```bash
# Baixar e instalar Docker Desktop
https://www.docker.com/products/docker-desktop/

# Verificar instalação
docker --version
docker-compose --version
```

### 2. PowerShell Execution Policy
```powershell
# Habilitar execução de scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Verificar política atual
Get-ExecutionPolicy
```

---

## 🐳 Docker para Windows

### Dockerfile.prod (Otimizado para Windows)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
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

### docker-compose.prod.yml
```yaml
version: '3.8'

services:
  app:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_PORT=8501
      - MLFLOW_TRACKING_URI=/app/mlruns
      - PYTHONPATH=/app
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - automl-network

  mlflow:
    image: python:3.11-slim
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=/app/mlruns
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns
    volumes:
      - ./mlruns:/app/mlruns
    command: >
      bash -c "
        pip install mlflow==2.9.2 &&
        mkdir -p /app/mlruns &&
        mlflow server 
        --host 0.0.0.0 
        --port 5000 
        --backend-store-uri /app/mlruns 
        --default-artifact-root /app/mlruns
      "
    restart: unless-stopped
    networks:
      - automl-network

networks:
  automl-network:
    driver: bridge
```

---

## 🚀 Métodos de Deploy

### Método 1: Script Automatizado (Recomendado)

#### deploy.bat (Simplificado)
```batch
@echo off
echo 🚀 Iniciando deploy do Multi-AutoML Interface...

# Verificar Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker não encontrado. Instale Docker Desktop.
    pause
    exit /b 1
)

# Build da imagem
echo 📦 Construindo imagem Docker...
docker build -f Dockerfile.prod -t multi-automl-interface:latest .

# Deploy com Docker Compose
echo 🚀 Iniciando containers...
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Aguardar inicialização
echo ⏳ Aguardando aplicacao iniciar (30 segundos)...
timeout /t 30 /nobreak

# Testar aplicação
echo 🧪 Testando aplicação...
curl -f http://localhost:8501 >nul 2>&1

echo.
echo 🎉 Deploy concluído!
echo.
echo 🌐 Acesse: http://localhost:8501
echo 📊 MLflow: http://localhost:5000
echo 📝 Logs: docker-compose -f docker-compose.prod.yml logs
echo 🛑 Parar: docker-compose -f docker-compose.prod.yml down
pause
```

#### Como usar:
```bash
# 1. Abrir PowerShell/CMD como Administrador
# 2. Navegar até o diretório do projeto
cd C:\Users\pedro\Downloads\Multi-AutoML-Interface

# 3. Executar script
deploy.bat
```

### Método 2: Manual Passo a Passo

#### Passo 1: Build da Imagem
```bash
# Construir imagem Docker
docker build -f Dockerfile.prod -t multi-automl-interface:latest .

# Verificar se imagem foi criada
docker images | findstr multi-automl-interface
```

#### Passo 2: Iniciar Containers
```bash
# Parar containers existentes
docker-compose -f docker-compose.prod.yml down

# Iniciar novos containers
docker-compose -f docker-compose.prod.yml up -d

# Verificar status
docker-compose -f docker-compose.prod.yml ps
```

#### Passo 3: Verificar Funcionamento
```bash
# Verificar logs
docker-compose -f docker-compose.prod.yml logs app

# Testar aplicação
curl http://localhost:8501

# Testar MLflow
curl http://localhost:5000
```

---

## 🔒 Configuração SSL em Windows

### Opção A: Nginx Reverse Proxy

#### nginx.conf
```nginx
events {
    worker_connections 1024;
}

http {
    upstream app {
        server app:8501;
    }

    upstream mlflow {
        server mlflow:5000;
    }

    # HTTP para App
    server {
        listen 80;
        server_name localhost;
        
        location / {
            proxy_pass http://app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }

    # HTTP para MLflow
    server {
        listen 80;
        server_name mlflow.localhost;
        
        location / {
            proxy_pass http://mlflow;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
```

#### docker-compose.ssl.yml
```yaml
version: '3.8'

services:
  app:
    build: 
      context: .
      dockerfile: Dockerfile.prod
    environment:
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
      - STREAMLIT_SERVER_PORT=8501
      - MLFLOW_TRACKING_URI=/app/mlruns
      - PYTHONPATH=/app
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    networks:
      - automl-network
    depends_on:
      - mlflow

  mlflow:
    image: python:3.11-slim
    environment:
      - MLFLOW_BACKEND_STORE_URI=/app/mlruns
      - MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns
    volumes:
      - ./mlruns:/app/mlruns
    command: >
      bash -c "
        pip install mlflow==2.9.2 &&
        mkdir -p /app/mlruns &&
        mlflow server 
        --host 0.0.0.0 
        --port 5000 
        --backend-store-uri /app/mlruns 
        --default-artifact-root /app/mlruns
      "
    restart: unless-stopped
    networks:
      - automl-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - app
    restart: unless-stopped
    networks:
      - automl-network

networks:
  automl-network:
    driver: bridge
```

### Opção B: Certificado Self-Signed

#### Gerar Certificado:
```bash
# Instalar OpenSSL
choco install openssl

# Gerar certificado
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configurar Streamlit para SSL
set STREAMLIT_SERVER_SSL_CERT_FILE=C:\path\to\cert.pem
set STREAMLIT_SERVER_SSL_KEY_FILE=C:\path\to\key.pem
```

---

## 🔧 Troubleshooting Windows

### Problema 1: Docker não encontrado
```bash
# Solução: Instalar Docker Desktop
https://www.docker.com/products/docker-desktop/

# Reiniciar serviço
net stop com.docker.service
net start com.docker.service
```

### Problema 2: Porta já em uso
```bash
# Verificar portas em uso
netstat -an | findstr :8501
netstat -an | findstr :5000

# Matar processo usando porta
taskkill /PID <PID> /F

# Ou mudar portas no docker-compose.yml
ports:
  - "8502:8501"  # Mudar porta externa
```

### Problema 3: Permissões negadas
```bash
# Executar como Administrador
# Ou configurar permissões
icacls . /grant Users:F /T
```

### Problema 4: Firewall bloqueando
```bash
# Liberar portas no Firewall
netsh advfirewall firewall add rule name="Streamlit" dir=in action=allow protocol=TCP localport=8501
netsh advfirewall firewall add rule name="MLflow" dir=in action=allow protocol=TCP localport=5000
```

### Problema 5: Docker Desktop não iniciando
```bash
# Reiniciar Docker Desktop
taskkill /F /IM "Docker Desktop.exe"
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Verificar serviço
sc query com.docker.service
```

---

## 📊 Monitoramento e Logs

### Verificar Logs em Tempo Real
```bash
# Logs de todos os containers
docker-compose -f docker-compose.prod.yml logs -f

# Logs apenas da aplicação
docker-compose -f docker-compose.prod.yml logs -f app

# Logs apenas do MLflow
docker-compose -f docker-compose.prod.yml logs -f mlflow
```

### Monitorar Recursos
```bash
# Uso de CPU e Memória
docker stats

# Informações detalhadas
docker inspect multi-automl-interface_app_1
```

### Backup Automático
```bash
# Criar script de backup
@echo off
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
mkdir backups\%timestamp%
xcopy mlruns backups\%timestamp%\mlruns /E /I /Y
xcopy data backups\%timestamp%\data /E /I /Y
echo Backup criado: backups\%timestamp%
```

---

## 🌐 Acesso Remoto

### Configurar para Acesso Externo
```bash
# Mudar bind para 0.0.0.0 no docker-compose.yml
ports:
  - "0.0.0.0:8501:8501"  # Aceitar conexões externas
  - "0.0.0.0:5000:5000"  # Aceitar conexões externas
```

### Configurar Router
```bash
# Port Forwarding no router
- Porta externa 8501 → Porta interna 8501
- Porta externa 5000 → Porta interna 5000
- IP do computador: ipconfig | findstr IPv4
```

### DNS Dinâmico
```bash
# Usar No-IP ou DuckDNS
# Instalar cliente de DNS dinâmico
# Configurar para atualizar automaticamente
```

---

## 🚀 Deploy Rápido

### Script Completo (deploy_completo.bat)
```batch
@echo off
setlocal enabledelayedexpansion

echo 🚀 Multi-AutoML Interface - Deploy Completo
echo ========================================

REM 1. Verificar dependências
echo 🔍 Verificando Docker Desktop...
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker não encontrado. Instale Docker Desktop.
    pause
    exit /b 1
)

REM 2. Criar diretórios necessários
echo 📁 Criando diretórios...
if not exist mlruns mkdir mlruns
if not exist data mkdir data
if not exist logs mkdir logs
if not exist backups mkdir backups

REM 3. Backup dos dados
echo 💾 Fazendo backup...
set timestamp=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%
if exist mlruns xcopy mlruns backups\%timestamp%\mlruns /E /I /Y >nul 2>&1

REM 4. Build da imagem
echo 📦 Construindo imagem Docker...
docker build -f Dockerfile.prod -t multi-automl-interface:latest . >build.log 2>&1
if %errorlevel% neq 0 (
    echo ❌ Falha no build. Verifique build.log
    type build.log
    pause
    exit /b 1
)

REM 5. Deploy
echo 🚀 Iniciando deploy...
docker-compose -f docker-compose.prod.yml down >deploy.log 2>&1
docker-compose -f docker-compose.prod.yml up -d >>deploy.log 2>&1
if %errorlevel% neq 0 (
    echo ❌ Falha no deploy. Verifique deploy.log
    type deploy.log
    pause
    exit /b 1
)

REM 6. Aguardar e testar
echo ⏳ Aguardando inicialização...
timeout /t 45 /nobreak

echo 🧪 Testando aplicação...
curl -f http://localhost:8501 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Aplicação está online!
) else (
    echo ⚠️ Aplicação pode estar iniciando...
    echo 📝 Verificando logs...
    docker-compose -f docker-compose.prod.yml logs app --tail=20
)

REM 7. Relatório final
echo.
echo 🎉 Deploy concluído com sucesso!
echo ========================================
echo 🌐 Aplicação: http://localhost:8501
echo 📊 MLflow: http://localhost:5000
echo 📝 Logs: docker-compose -f docker-compose.prod.yml logs
echo 🛑 Parar: docker-compose -f docker-compose.prod.yml down
echo 🔄 Reiniciar: docker-compose -f docker-compose.prod.yml restart
echo 💾 Backup: backups\%timestamp%
echo ========================================
echo.
pause
```

---

## 📞 Suporte Windows

### Comandos Úteis
```bash
# Verificar containers ativos
docker ps

# Verificar todas as imagens
docker images

# Limpar recursos não utilizados
docker system prune -f

# Reiniciar Docker Desktop
Restart-Service -Name com.docker.service

# Verificar portas
netstat -an | findstr LISTENING
```

### Logs do Sistema
```bash
# Logs do Docker Desktop
%APPDATA%\Docker\log.txt

# Logs do Docker Daemon
%PROGRAMDATA%\Docker\log\daemon.log
```

---

**Este guia cobre tudo necessário para deploy em Windows!** 🚀

**Para suporte adicional, consulte o README_DEPLOYMENT.md principal.** 📚
