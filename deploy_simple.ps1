# 🚀 Multi-AutoML Interface - Simple Windows Deployment
# Script simplificado para deploy em Windows

Write-Host "🚀 Iniciando deploy do Multi-AutoML Interface..." -ForegroundColor Green

# Verificar Docker
try {
    $dockerVersion = docker --version
    Write-Host "✅ Docker encontrado: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker não encontrado. Instale Docker Desktop." -ForegroundColor Red
    exit 1
}

# Build da imagem
Write-Host "📦 Construindo imagem Docker..." -ForegroundColor Blue
try {
    docker build -f Dockerfile.prod -t multi-automl-interface:latest .
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Imagem construída com sucesso!" -ForegroundColor Green
    } else {
        Write-Host "❌ Falha ao construir imagem" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Erro no build: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Deploy com Docker Compose
Write-Host "🚀 Iniciando containers..." -ForegroundColor Blue
try {
    docker-compose -f docker-compose.prod.yml down
    docker-compose -f docker-compose.prod.yml up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Deploy iniciado com sucesso!" -ForegroundColor Green
    } else {
        Write-Host "❌ Falha ao iniciar containers" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Erro no deploy: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Aguardar inicialização
Write-Host "⏳ Aguardando aplicacao iniciar (30 segundos)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Testar aplicação
Write-Host "🧪 Testando aplicação..." -ForegroundColor Blue
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8501" -UseBasicParsing -TimeoutSec 10
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Aplicação está respondendo!" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Aplicação pode estar iniciando..." -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Aplicação pode ainda estar iniciando..." -ForegroundColor Yellow
}

# Relatório final
Write-Host "" -ForegroundColor White
Write-Host "🎉 Deploy concluído!" -ForegroundColor Green
Write-Host "" -ForegroundColor White
Write-Host "🌐 Acesse: http://localhost:8501" -ForegroundColor Cyan
Write-Host "📊 MLflow: http://localhost:5000" -ForegroundColor Cyan
Write-Host "📝 Logs: docker-compose -f docker-compose.prod.yml logs" -ForegroundColor Cyan
Write-Host "🛑 Parar: docker-compose -f docker-compose.prod.yml down" -ForegroundColor Cyan
