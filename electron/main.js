const { app, BrowserWindow, Menu, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// Mantém referência global da janela
let mainWindow;
let streamlitProcess;
let pythonPath;

// The Streamlit server this window is allowed to display.
const APP_PORT = 8501;
const APP_URLS = [`http://127.0.0.1:${APP_PORT}`, `http://localhost:${APP_PORT}`];

// Directory holding the interpreter + dependencies shipped inside the installer
// (built by scripts/prepare_python_runtime.js).
const RUNTIME_ROOT = app.isPackaged
    ? path.join(process.resourcesPath, 'runtime')
    : path.join(__dirname, '..', 'runtime');

// Source of the Streamlit app (app.py, src/).
const APP_ROOT = path.join(__dirname, '..');

function resolvePython() {
    // Prefer the bundled runtime; fall back to the system interpreter for source checkouts.
    const manifestPath = path.join(RUNTIME_ROOT, 'runtime-manifest.json');
    if (fs.existsSync(manifestPath)) {
        try {
            const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
            const exe = path.join(RUNTIME_ROOT, manifest.interpreter);
            if (fs.existsSync(exe)) return { exe, source: 'bundled' };
        } catch (error) {
            console.error('runtime-manifest.json is unreadable:', error);
        }
    }
    return { exe: process.platform === 'win32' ? 'python' : 'python3', source: 'system' };
}

// app.py writes mlruns/, models/ and data_lake/ relative to its working directory. For the
// installed app that would be Program Files, which normal users cannot write to, so runs
// happen in a per-user workspace while the sources stay in the app folder.
function workspaceRoot() {
    const dir = app.isPackaged ? path.join(app.getPath('userData'), 'workspace') : APP_ROOT;
    fs.mkdirSync(dir, { recursive: true });
    return dir;
}

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
                        // The desktop app records runs in the local ./mlruns file store.
                        // A tracking UI only exists if a server is running, so open the
                        // configured one and explain the rest instead of a dead localhost:5000.
                        const server = process.env.MLFLOW_TRACKING_URI;
                        if (server && /^https?:\/\//i.test(server)) {
                            shell.openExternal(server);
                        } else {
                            dialog.showMessageBox(mainWindow, {
                                type: 'info',
                                title: 'MLflow',
                                message: 'Runs are stored in the local ./mlruns folder',
                                detail: 'Start the tracking server with "docker compose up" and set MLFLOW_TRACKING_URI to open its UI here.'
                            });
                        }
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
                            message: `Multi-AutoML Desktop v${app.getVersion()}`,
                            detail: 'Interface desktop para AutoML com AutoGluon, FLAML e H2O\\n\\nDesenvolvido com ❤️ usando Electron e Streamlit'
                        });
                    }
                },
                {
                    label: 'Documentação',
                    click: () => {
                        shell.openExternal('https://github.com/PedroM2626/Multi-AutoML-Interface');
                    }
                }
            ]
        }
    ];

    const menu = Menu.buildFromTemplate(template);
    Menu.setApplicationMenu(menu);

    // Carregar a aplicação Streamlit
    const loadUrlWithRetry = (retries = 0) => {
        mainWindow.loadURL(APP_URLS[0]).catch((err) => {
            console.log(`Server not ready, retrying... (${retries})`);
            if (retries < 20) {
                setTimeout(() => loadUrlWithRetry(retries + 1), 1000);
            } else {
                mainWindow.loadFile(path.join(__dirname, '..', 'error_loading.html')).catch(e => {
                    console.error('Failed to load error_loading.html:', e);
                });
            }
        });
    };
    loadUrlWithRetry();

    // Mostrar janela quando estiver pronta
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        mainWindow.center();
    });

    // Open external links in the system browser, but only for http(s) targets:
    // file:// and custom schemes handed to openExternal can launch local programs.
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        if (/^https?:\/\//i.test(url)) {
            shell.openExternal(url);
        }
        return { action: 'deny' };
    });

    // Keep the top frame on the local Streamlit app; a redirect away from it would
    // still hand the page the APIs exposed by preload.js.
    mainWindow.webContents.on('will-navigate', (event, url) => {
        if (!APP_URLS.some((allowed) => url.startsWith(allowed))) {
            event.preventDefault();
            if (/^https?:\/\//i.test(url)) {
                shell.openExternal(url);
            }
        }
    });

    // Fechar janela
    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

// Iniciar Streamlit
function startStreamlit() {
    const { spawn, execFileSync } = require('child_process');
    const resolved = resolvePython();
    pythonPath = resolved.exe;
    const workspace = workspaceRoot();

    // Fail with something actionable instead of a generic "Streamlit failed to start":
    // the bundled runtime is only usable if the app's own dependencies import cleanly.
    try {
        execFileSync(pythonPath, ['-c', 'import streamlit, mlflow'], { stdio: 'pipe', timeout: 120000 });
    } catch (error) {
        const detail = resolved.source === 'bundled'
            ? `O runtime incluído no instalador (${pythonPath}) não está funcionando: as bibliotecas do app não importam.`
            : `Nenhum runtime embutido foi encontrado e o Python do sistema (${pythonPath}) não tem as dependências do app.`;
        console.error('Python environment check failed:', String(error).slice(0, 300));
        dialog.showErrorBox(
            'Ambiente Python indisponível',
            `${detail}\n\nReinstale o aplicativo ou execute "pip install -r requirements.txt" no interpretador usado.`
        );
        app.quit();
        return;
    }

    // Start Streamlit. CORS is left at its secure default even though the server binds
    // to loopback: with CORS disabled, any page open in the user's browser could read
    // and post to http://127.0.0.1:<port>.
    streamlitProcess = spawn(pythonPath, [
        '-m', 'streamlit', 'run', path.join(APP_ROOT, 'app.py'),
        '--server.port', String(APP_PORT),
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.address', '127.0.0.1'
    ], {
        cwd: workspace,
        // app.py lives in the app folder, the process runs in the writable workspace, so
        // the sources have to stay importable (from src.xxx import ...).
        env: { ...process.env, PYTHONPATH: APP_ROOT },
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
