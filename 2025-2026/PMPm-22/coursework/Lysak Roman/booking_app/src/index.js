import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './components/App';
import { PublicClientApplication } from "@azure/msal-browser";
import { MsalProvider } from "@azure/msal-react";
import { msalConfig } from "./authConfig";
import { BookingsProvider } from './contexts/BookingsContext';
import { LLMProvider } from './contexts/LLMContext';

const msalInstance = new PublicClientApplication(msalConfig);


const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <MsalProvider instance={msalInstance}>
      <BookingsProvider>
        <LLMProvider>
          <App />
        </LLMProvider>
      </BookingsProvider>
    </MsalProvider>
  </React.StrictMode>
);

