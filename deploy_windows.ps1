# 🚀 Multi-AutoML Interface - Windows Deployment Script
# Este script automatiza o deploy completo da aplicação em Windows

param(
    [string]$Environment = "production",
    [switch]$SkipTests = $false,
    [switch]$SkipBackup = $false
)

# Cores para output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Funções de log
function Write-LogInfo {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor $Colors.Blue
}

function Write-LogSuccess {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor $Colors.Green
}

function Write-LogWarning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor $Colors.Yellow
}

function Write-LogError {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor $Colors.Red
}

# Verificar dependências
function Test-Dependencies {
    Write-LogInfo "Verificando dependências..."
    
    # Verificar Docker Desktop
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Docker encontrado: $dockerVersion"
        } else {
            Write-LogError "Docker não encontrado. Por favor, instale o Docker Desktop."
            exit 1
        }
    } catch {
        Write-LogError "Docker não encontrado. Por favor, instale o Docker Desktop."
        exit 1
    }
    
    # Verificar Docker Compose
    try {
        $composeVersion = docker-compose --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Docker Compose encontrado: $composeVersion"
        } else {
            Write-LogError "Docker Compose não encontrado. Por favor, instale o Docker Compose."
            exit 1
        }
    } catch {
        Write-LogError "Docker Compose não encontrado. Por favor, instale o Docker Compose."
        exit 1
    }
    
    Write-LogSuccess "Dependências verificadas com sucesso!"
}

# Backup dos dados
function Backup-Data {
    if ($SkipBackup) {
        Write-LogWarning "Backup pulado por parâmetro"
        return
    }
    
    Write-LogInfo "Fazendo backup dos dados existentes..."
    
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    
    if (Test-Path "mlruns") {
        Copy-Item -Path "mlruns" -Destination "mlruns.backup.$timestamp" -Recurse -Force
        Write-LogSuccess "Backup de mlruns criado: mlruns.backup.$timestamp"
    }
    
    if (Test-Path "data") {
        Copy-Item -Path "data" -Destination "data.backup.$timestamp" -Recurse -Force
        Write-LogSuccess "Backup de data criado: data.backup.$timestamp"
    }
}

# Build da imagem Docker
function Build-DockerImage {
    Write-LogInfo "Construindo imagem Docker..."
    
    try {
        $buildResult = docker build -f Dockerfile.prod -t multi-automl-interface:latest . 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Imagem Docker construída com sucesso!"
        } else {
            Write-LogError "Falha ao construir imagem Docker"
            Write-LogError "Erro: $buildResult"
            exit 1
        }
    } catch {
        Write-LogError "Exceção ao construir imagem Docker: $($_.Exception.Message)"
        exit 1
    }
}

# Deploy com Docker Compose
function Deploy-WithCompose {
    Write-LogInfo "Iniciando deploy com Docker Compose..."
    
    try {
        # Parar containers existentes
        docker-compose -f docker-compose.prod.yml down 2>$null
        
        # Iniciar novos containers
        $deployResult = docker-compose -f docker-compose.prod.yml up -d 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-LogSuccess "Deploy iniciado com sucesso!"
        } else {
            Write-LogError "Falha ao iniciar containers"
            Write-LogError "Erro: $deployResult"
            exit 1
        }
    } catch {
        Write-LogError "Exceção durante deploy: $($_.Exception.Message)"
        exit 1
    }
}

# Verificar status dos containers
function Test-Containers {
    Write-LogInfo "Verificando status dos containers..."
    
    # Aguardar containers iniciarem
    Start-Sleep -Seconds 10
    
    try {
        # Verificar se containers estão rodando
        $containerStatus = docker-compose -f docker-compose.prod.yml ps 2>&1
        if ($containerStatus -match "Up") {
            Write-LogSuccess "Containers estão rodando corretamente!"
        } else {
            Write-LogError "Containers não estão rodando corretamente"
            docker-compose -f docker-compose.prod.yml logs
            exit 1
        }
    } catch {
        Write-LogError "Exceção ao verificar containers: $($_.Exception.Message)"
        exit 1
    }
}

# Testar aplicação
function Test-Application {
    Write-LogInfo "Testando aplicação..."
    
    # Aguardar aplicação iniciar
    Write-LogInfo "Aguardando aplicação iniciar (30 segundos)..."
    Start-Sleep -Seconds 30
    
    # Testar se a aplicação está respondendo
    try {
        $appResponse = Invoke-WebRequest -Uri "http://localhost:8501/_stcore/health" -UseBasicParsing -TimeoutSec 10 2>$null
        if ($appResponse.StatusCode -eq 200) {
            Write-LogSuccess "Aplicação está respondendo corretamente!"
        } else {
            Write-LogWarning "Aplicação pode ainda estar iniciando... Status: $($appResponse.StatusCode)"
        }
    } catch {
        Write-LogWarning "Aplicação pode ainda estar iniciando..."
        Write-LogInfo "Verificando logs..."
        docker-compose -f docker-compose.prod.yml logs app
    }
    
    # Testar MLflow
    try {
        $mlflowResponse = Invoke-WebRequest -Uri "http://localhost:5000" -UseBasicParsing -TimeoutSec 10 2>$null
        if ($mlflowResponse.StatusCode -eq 200) {
            Write-LogSuccess "MLflow está respondendo corretamente!"
        } else {
            Write-LogWarning "MLflow pode ainda estar iniciando... Status: $($mlflowResponse.StatusCode)"
        }
    } catch {
        Write-LogWarning "MLflow pode ainda estar iniciando..."
    }
}

# Gerar relatório de deploy
function New-DeployReport {
    Write-LogInfo "Gerando relatório de deploy..."
    
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $hostname = hostname
    $dockerVersion = docker --version
    
    $report = @"
🚀 Multi-AutoML Interface - Deployment Report
📅 Data: $timestamp
🖥️  Host: $hostname
🐳 Docker: $dockerVersion
📦 Imagem: multi-automl-interface:latest
🌐 App URL: http://localhost:8501
📊 MLflow URL: http://localhost:5000
📝 Logs: docker-compose -f docker-compose.prod.yml logs
🛑 Parar: docker-compose -f docker-compose.prod.yml down
🔄 Reiniciar: docker-compose -f docker-compose.prod.yml restart
🗂️  Backup: Criado automaticamente
"@
    
    $report | Out-File -FilePath "deploy_report.txt" -Encoding UTF8
    Write-LogSuccess "Relatório de deploy gerado: deploy_report.txt"
}

# Função principal
function Main {
    Write-LogInfo "🎯 Multi-AutoML Interface Deployment Script (Windows)"
    Write-Host "========================================" -ForegroundColor $Colors.White
    
    # Executar passos do deploy
    Test-Dependencies
    Backup-Data
    Build-DockerImage
    Deploy-WithCompose
    Test-Containers
    Test-Application
    New-DeployReport
    
    Write-Host "" -ForegroundColor $Colors.White
    Write-LogSuccess "🎉 Deploy concluído com sucesso!"
    Write-Host "" -ForegroundColor $Colors.White
    Write-LogInfo "🌐 Acesse a aplicação em: http://localhost:8501"
    Write-LogInfo "📊 Acesse o MLflow em: http://localhost:5000"
    Write-LogInfo "📝 Veja os logs com: docker-compose -f docker-compose.prod.yml logs"
    Write-LogInfo "🛑 Para parar: docker-compose -f docker-compose.prod.yml down"
    Write-Host "" -ForegroundColor $Colors.White
    Write-LogInfo "📋 Relatório completo salvo em: deploy_report.txt"
}

# Executar função principal
try {
    Main
} catch {
    Write-LogError "Erro fatal durante deploy: $($_.Exception.Message)"
    exit 1
}
