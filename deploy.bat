@echo off
REM 🚀 Multi-AutoML Interface - Simple Windows Deployment

echo 🚀 Iniciando deploy do Multi-AutoML Interface...

REM Verificar Docker
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker não encontrado. Instale Docker Desktop.
    pause
    exit /b 1
)
echo ✅ Docker encontrado

REM Build da imagem
echo 📦 Construindo imagem Docker...
docker build -f Dockerfile.prod -t multi-automl-interface:latest .
if %errorlevel% neq 0 (
    echo ❌ Falha ao construir imagem
    pause
    exit /b 1
)
echo ✅ Imagem construída com sucesso!

REM Deploy com Docker Compose
echo 🚀 Iniciando containers...
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d
if %errorlevel% neq 0 (
    echo ❌ Falha ao iniciar containers
    pause
    exit /b 1
)
echo ✅ Deploy iniciado com sucesso!

REM Aguardar inicialização
echo ⏳ Aguardando aplicacao iniciar (30 segundos)...
timeout /t 30 /nobreak

REM Testar aplicação
echo 🧪 Testando aplicação...
curl -f http://localhost:8501 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Aplicação está respondendo!
) else (
    echo ⚠️ Aplicação pode estar iniciando...
)

REM Relatório final
echo.
echo 🎉 Deploy concluído!
echo.
echo 🌐 Acesse: http://localhost:8501
echo 📊 MLflow: http://localhost:5000
echo 📝 Logs: docker-compose -f docker-compose.prod.yml logs
echo 🛑 Parar: docker-compose -f docker-compose.prod.yml down
echo.
pause
