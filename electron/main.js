const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Mantém referência global da janela
let mainWindow;
let streamlitProcess;
let pythonPath;

function createWindow() {
    // Criar janela principal
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        minWidth: 1200,
        minHeight: 800,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            enableRemoteModule: false,
            preload: path.join(__dirname, 'preload.js')
        },
        icon: path.join(__dirname, 'assets', 'icon.png'),
        show: false, // Esconder até estar pronto
        titleBarStyle: 'default'
    });

    // Menu da aplicação
    const template = [
        {
            label: 'Arquivo',
            submenu: [
                {
                    label: 'Abrir Dataset',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => {
                        dialog.showOpenDialog(mainWindow, {
                            properties: ['openFile'],
                            filters: [
                                { name: 'CSV Files', extensions: ['csv'] },
                                { name: 'Excel Files', extensions: ['xlsx', 'xls'] },
                                { name: 'All Files', extensions: ['*'] }
                            ]
                        }).then(result => {
                            if (!result.canceled) {
                                mainWindow.webContents.send('file-selected', result.filePaths[0]);
                            }
                        });
                    }
                },
                { type: 'separator' },
                {
                    label: 'Sair',
                    accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
                    click: () => {
                        app.quit();
                    }
                }
            ]
        },
        {
            label: 'Editar',
            submenu: [
                { label: 'Desfazer', accelerator: 'CmdOrCtrl+Z', role: 'undo' },
                { label: 'Refazer', accelerator: 'Shift+CmdOrCtrl+Z', role: 'redo' },
                { type: 'separator' },
                { label: 'Copiar', accelerator: 'CmdOrCtrl+C', role: 'copy' },
                { label: 'Colar', accelerator: 'CmdOrCtrl+V', role: 'paste' }
            ]
        },
        {
            label: 'Ferramentas',
            submenu: [
                {
                    label: 'Abrir MLflow',
                    click: () => {
                        shell.openExternal('http://localhost:5000');
                    }
                },
                {
                    label: 'Limpar Cache',
                    click: () => {
                        mainWindow.webContents.send('clear-cache');
                    }
                },
                { type: 'separator' },
                {
                    label: 'Developer Tools',
                    accelerator: 'F12',
                    click: () => {
                        mainWindow.webContents.toggleDevTools();
                    }
                }
            ]
        },
        {
            label: 'Ajuda',
            submenu: [
                {
                    label: 'Sobre',
                    click: () => {
                        dialog.showMessageBox(mainWindow, {
                            type: 'info',
                            title: 'Sobre Multi-AutoML Desktop',
                            message: 'Multi-AutoML Desktop v1.0.0',
                            detail: 'Interface desktop para AutoML com AutoGluon, FLAML e H2O\\n\\nDesenvolvido com ❤️ usando Electron e Streamlit'
                        });
                    }
                },
                {
                    label: 'Documentação',
                    click: () => {
                        shell.openExternal('https://github.com/seu-usuario/multi-automl-interface');
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    // Carregar a aplicação Streamlit
    mainWindow.loadURL('http://localhost:8501');

    // Mostrar janela quando estiver pronta
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.center();
    });

    // Abrir links externos no navegador
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    // Fechar janela
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Iniciar Streamlit
function startStreamlit() {
    const platform = process.platform;
    
    // Encontrar Python
    if (platform === 'win32') {
        pythonPath = 'python';
    } else if (platform === 'darwin') {
        pythonPath = 'python3';
    } else {
        pythonPath = 'python3';
    }

    // Verificar se Python existe
    const { spawn } = require('child_process');
    const pythonCheck = spawn(pythonPath, ['--version']);
    
    pythonCheck.on('error', (error) => {
        console.error('Python não encontrado:', error);
        dialog.showErrorBox('Erro', 'Python não encontrado. Por favor, instale Python 3.8+ para continuar.');
        app.quit();
    });

    // Iniciar Streamlit
    streamlitProcess = spawn(pythonPath, [
        '-m', 'streamlit', 'run', 'app.py',
        '--server.port', '8501',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--browser.gatherUsageStats', 'false'
    ], {
        cwd: path.join(__dirname, '..'),
        stdio: 'pipe'
    });

    streamlitProcess.stdout.on('data', (data) => {
        console.log(`Streamlit: ${data}`);
    });

    streamlitProcess.stderr.on('data', (data) => {
        console.error(`Streamlit Error: ${data}`);
    });

    streamlitProcess.on('close', (code) => {
        console.log(`Streamlit process exited with code ${code}`);
        if (code !== 0) {
            dialog.showErrorBox('Erro', 'Streamlit falhou ao iniciar. Verifique o console para detalhes.');
        }
    });
}

// IPC handlers
ipcMain.handle('get-app-version', () => {
    return app.getVersion();
});

ipcMain.handle('get-python-path', () => {
    return pythonPath;
});

ipcMain.handle('restart-streamlit', async () => {
    if (streamlitProcess) {
        streamlitProcess.kill();
        await new Promise(resolve => setTimeout(resolve, 2000));
        startStreamlit();
        return true;
    }
    return false;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
});

// Eventos da aplicação
app.whenReady().then(() => {
    startStreamlit();
    
    // Esperar Streamlit iniciar
    setTimeout(() => {
        createWindow();
    }, 3000);

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});

app.on('window-all-closed', () => {
    // Fechar Streamlit quando todas as janelas fecharem
    if (streamlitProcess) {
        streamlitProcess.kill();
    }
    
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('before-quit', () => {
    // Limpar processos
    if (streamlitProcess) {
        streamlitProcess.kill();
    }
});

// Security: desabilitar algumas features por segurança
app.on('web-contents-created', (event, contents) => {
    contents.on('new-window', (event, navigationUrl) => {
        event.preventDefault();
        shell.openExternal(navigationUrl);
    });
});
