const { contextBridge, ipcRenderer } = require('electron');

// Expor APIs seguras para o renderer process
contextBridge.exposeInMainWorld('electronAPI', {
    // App info
    getAppVersion: () => ipcRenderer.invoke('get-app-version'),
    getPythonPath: () => ipcRenderer.invoke('get-python-path'),
    
    // File operations
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    
    // Streamlit control
    restartStreamlit: () => ipcRenderer.invoke('restart-streamlit'),
    
    // Events
    onFileSelected: (callback) => ipcRenderer.on('file-selected', callback),
    onClearCache: (callback) => ipcRenderer.on('clear-cache', callback),
    
    // Remove listeners
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});

// Expose some useful information to the renderer
window.electron = {
    isElectron: true,
    platform: process.platform,
    version: process.versions.electron
};
