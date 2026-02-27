#!/bin/bash

# 🚀 Multi-AutoML Interface - Deployment Script
# Este script automatiza o deploy completo da aplicação

set -e  # Exit on any error

echo "🚀 Iniciando deployment do Multi-AutoML Interface..."

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Funções de log
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Verificar dependências
check_dependencies() {
    log_info "Verificando dependências..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker não encontrado. Por favor, instale o Docker."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose não encontrado. Por favor, instale o Docker Compose."
        exit 1
    fi
    
    log_success "Dependências verificadas com sucesso!"
}

# Backup dos dados
backup_data() {
    log_info "Fazendo backup dos dados existentes..."
    
    if [ -d "mlruns" ]; then
        cp -r mlruns mlruns.backup.$(date +%Y%m%d_%H%M%S)
        log_success "Backup de mlruns criado"
    fi
    
    if [ -d "data" ]; then
        cp -r data data.backup.$(date +%Y%m%d_%H%M%S)
        log_success "Backup de data criado"
    fi
}

# Build da imagem Docker
build_image() {
    log_info "Construindo imagem Docker..."
    
    docker build -f Dockerfile.prod -t multi-automl-interface:latest .
    
    if [ $? -eq 0 ]; then
        log_success "Imagem Docker construída com sucesso!"
    else
        log_error "Falha ao construir imagem Docker"
        exit 1
    fi
}

# Deploy com Docker Compose
deploy_with_compose() {
    log_info "Iniciando deploy com Docker Compose..."
    
    # Parar containers existentes
    docker-compose -f docker-compose.prod.yml down || true
    
    # Iniciar novos containers
    docker-compose -f docker-compose.prod.yml up -d
    
    if [ $? -eq 0 ]; then
        log_success "Deploy iniciado com sucesso!"
    else
        log_error "Falha ao iniciar containers"
        exit 1
    fi
}

# Verificar status dos containers
check_containers() {
    log_info "Verificando status dos containers..."
    
    sleep 10  # Aguardar containers iniciarem
    
    # Verificar se containers estão rodando
    if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        log_success "Containers estão rodando corretamente!"
    else
        log_error "Containers não estão rodando corretamente"
        docker-compose -f docker-compose.prod.yml logs
        exit 1
    fi
}

# Testar aplicação
test_application() {
    log_info "Testando aplicação..."
    
    # Aguardar aplicação iniciar
    sleep 30
    
    # Testar se a aplicação está respondendo
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        log_success "Aplicação está respondendo corretamente!"
    else
        log_warning "Aplicação pode ainda estar iniciando..."
        log_info "Verificando logs..."
        docker-compose -f docker-compose.prod.yml logs app
    fi
    
    # Testar MLflow
    if curl -f http://localhost:5000 > /dev/null 2>&1; then
        log_success "MLflow está respondendo corretamente!"
    else
        log_warning "MLflow pode ainda estar iniciando..."
    fi
}

# Gerar relatório de deploy
generate_report() {
    log_info "Gerando relatório de deploy..."
    
    cat > deploy_report.txt << EOF
🚀 Multi-AutoML Interface - Deployment Report
📅 Data: $(date)
🖥️  Host: $(hostname)
🐳 Docker: $(docker --version)
📦 Imagem: multi-automl-interface:latest
🌐 App URL: http://localhost:8501
📊 MLflow URL: http://localhost:5000
📝 Logs: docker-compose -f docker-compose.prod.yml logs
🛑 Parar: docker-compose -f docker-compose.prod.yml down
🔄 Reiniciar: docker-compose -f docker-compose.prod.yml restart
EOF
    
    log_success "Relatório de deploy gerado: deploy_report.txt"
}

# Função principal
main() {
    log_info "🎯 Multi-AutoML Interface Deployment Script"
    echo "========================================"
    
    # Executar passos do deploy
    check_dependencies
    backup_data
    build_image
    deploy_with_compose
    check_containers
    test_application
    generate_report
    
    echo ""
    log_success "🎉 Deploy concluído com sucesso!"
    echo ""
    log_info "🌐 Acesse a aplicação em: http://localhost:8501"
    log_info "📊 Acesse o MLflow em: http://localhost:5000"
    log_info "📝 Veja os logs com: docker-compose -f docker-compose.prod.yml logs"
    log_info "🛑 Para parar: docker-compose -f docker-compose.prod.yml down"
    echo ""
    log_info "📋 Relatório completo salvo em: deploy_report.txt"
}

# Executar função principal
main "$@"
