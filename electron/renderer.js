// Renderer process - Interface enhancements for Electron

class ElectronInterface {
    constructor() {
        this.isElectron = window.electron?.isElectron || false;
        this.init();
    }

    init() {
        if (!this.isElectron) return;

        // Adicionar estilos específicos para desktop
        this.addDesktopStyles();
        
        // Adicionar funcionalidades desktop
        this.addDesktopFeatures();
        
        // Adicionar atalhos de teclado
        this.addKeyboardShortcuts();
        
        // Adicionar indicador de modo desktop
        this.addDesktopIndicator();
    }

    addDesktopStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .desktop-mode {
                --desktop-primary: #2563eb;
                --desktop-secondary: #64748b;
            }
            
            .desktop-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1rem;
                margin: -1rem -1rem 1rem -1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .desktop-title {
                font-size: 1.5rem;
                font-weight: bold;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .desktop-controls {
                display: flex;
                gap: 1rem;
                align-items: center;
            }
            
            .desktop-btn {
                background: rgba(255, 255, 255, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.3);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            
            .desktop-btn:hover {
                background: rgba(255, 255, 255, 0.3);
                transform: translateY(-1px);
            }
            
            .desktop-indicator {
                position: fixed;
                top: 10px;
                right: 10px;
                background: #10b981;
                color: white;
                padding: 0.25rem 0.75rem;
                border-radius: 1rem;
                font-size: 0.75rem;
                font-weight: bold;
                z-index: 9999;
            }
            
            .desktop-status {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-size: 0.875rem;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 2s infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
        `;
        document.head.appendChild(style);
    }

    addDesktopFeatures() {
        // Esperar a página carregar completamente
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.createDesktopUI());
        } else {
            this.createDesktopUI();
        }
    }

    createDesktopUI() {
        // Adicionar header desktop
        const header = document.createElement('div');
        header.className = 'desktop-header';
        header.innerHTML = `
            <div class="desktop-title">
                🚀 Multi-AutoML Desktop
            </div>
            <div class="desktop-controls">
                <div class="desktop-status">
                    <div class="status-dot"></div>
                    <span>Streamlit Conectado</span>
                </div>
                <button class="desktop-btn" onclick="window.electronInterface.openMLflow()">
                    📊 MLflow
                </button>
                <button class="desktop-btn" onclick="window.electronInterface.clearCache()">
                    🗑️ Limpar Cache
                </button>
                <button class="desktop-btn" onclick="window.electronInterface.showAbout()">
                    ℹ️ Sobre
                </button>
            </div>
        `;
        
        // Inserir header no início do body
        const firstElement = document.body.firstElementChild;
        if (firstElement) {
            document.body.insertBefore(header, firstElement);
        } else {
            document.body.appendChild(header);
        }

        // Adicionar indicador de modo desktop
        const indicator = document.createElement('div');
        indicator.className = 'desktop-indicator';
        indicator.textContent = 'Desktop Mode';
        document.body.appendChild(indicator);
    }

    addKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+O para abrir arquivo
            if ((e.ctrlKey || e.metaKey) && e.key === 'o') {
                e.preventDefault();
                this.openFile();
            }
            
            // Ctrl+Shift+R para reiniciar Streamlit
            if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'R') {
                e.preventDefault();
                this.restartStreamlit();
            }
            
            // F11 para toggle fullscreen
            if (e.key === 'F11') {
                e.preventDefault();
                this.toggleFullscreen();
            }
        });
    }

    addDesktopIndicator() {
        // Adicionar informações sobre o modo desktop na sidebar
        const observer = new MutationObserver(() => {
            const sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (sidebar && !sidebar.querySelector('.desktop-info')) {
                const info = document.createElement('div');
                info.className = 'desktop-info';
                info.style.cssText = `
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1rem;
                    margin: 1rem;
                    border-radius: 0.5rem;
                    text-align: center;
                `;
                info.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 0.5rem;">💻 Desktop Mode</div>
                    <div style="font-size: 0.875rem; opacity: 0.9;">
                        Multi-AutoML Desktop v${window.electron?.version || '1.0.0'}
                    </div>
                    <div style="font-size: 0.75rem; margin-top: 0.5rem; opacity: 0.8;">
                        Platform: ${window.electron?.platform || 'Unknown'}
                    </div>
                `;
                sidebar.appendChild(info);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Métodos de interface
    async openFile() {
        try {
            const result = await window.electronAPI.showOpenDialog({
                properties: ['openFile'],
                filters: [
                    { name: 'CSV Files', extensions: ['csv'] },
                    { name: 'Excel Files', extensions: ['xlsx', 'xls'] },
                    { name: 'All Files', extensions: ['*'] }
                ]
            });
            
            if (!result.canceled && result.filePaths.length > 0) {
                // Encontrar o uploader de arquivo e simular upload
                const fileInput = document.querySelector('input[type="file"]');
                if (fileInput) {
                    // Criar um arquivo fake para o input
                    const file = new File([''], result.filePaths[0].split('\\').pop());
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    fileInput.files = dataTransfer.files;
                    
                    // Disparar evento change
                    const event = new Event('change', { bubbles: true });
                    fileInput.dispatchEvent(event);
                }
            }
        } catch (error) {
            console.error('Erro ao abrir arquivo:', error);
        }
    }

    async restartStreamlit() {
        try {
            const success = await window.electronAPI.restartStreamlit();
            if (success) {
                // Mostrar mensagem de sucesso
                this.showNotification('Streamlit reiniciado com sucesso!', 'success');
                // Recarregar página após 2 segundos
                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                this.showNotification('Falha ao reiniciar Streamlit', 'error');
            }
        } catch (error) {
            console.error('Erro ao reiniciar Streamlit:', error);
            this.showNotification('Erro ao reiniciar Streamlit', 'error');
        }
    }

    toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    }

    openMLflow() {
        window.open('http://localhost:5000', '_blank');
    }

    async clearCache() {
        try {
            // Limpar localStorage
            localStorage.clear();
            
            // Limpar sessionStorage
            sessionStorage.clear();
            
            // Enviar mensagem para o main process
            window.electronAPI.onClearCache(() => {});
            
            this.showNotification('Cache limpo com sucesso!', 'success');
            
            // Recarregar página após 1 segundo
            setTimeout(() => {
                window.location.reload();
            }, 1000);
        } catch (error) {
            console.error('Erro ao limpar cache:', error);
            this.showNotification('Erro ao limpar cache', 'error');
        }
    }

    async showAbout() {
        try {
            const version = await window.electronAPI.getAppVersion();
            const platform = window.electron?.platform || 'Unknown';
            
            this.showNotification(
                `Multi-AutoML Desktop v${version}\nPlatform: ${platform}\n\nPowered by Electron + Streamlit`,
                'info',
                5000
            );
        } catch (error) {
            console.error('Erro ao mostrar sobre:', error);
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        // Criar elemento de notificação
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            z-index: 10000;
            max-width: 300px;
            white-space: pre-line;
            animation: slideIn 0.3s ease;
        `;
        notification.textContent = message;
        
        // Adicionar animação
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideIn {
                from {
                    transform: translateX(100%);
                    opacity: 0;
                }
                to {
                    transform: translateX(0);
                    opacity: 1;
                }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(notification);
        
        // Remover após duration
        setTimeout(() => {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, duration);
    }
}

// Inicializar interface quando disponível
window.electronInterface = new ElectronInterface();
