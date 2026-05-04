import React from 'react';
import ReactDOM from 'react-dom/client';
import { PublicClientApplication } from '@azure/msal-browser';
import { msalConfig } from './authConfig';
import App from './App';
import './index.css';

// Export MSAL instance so api.ts and useSignalR can use it without circular deps
export const msalInstance = new PublicClientApplication(msalConfig);

// Handle redirect promise (needed when using redirect flow)
msalInstance.initialize().then(() => {
  msalInstance.handleRedirectPromise().catch(console.error);

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
});
