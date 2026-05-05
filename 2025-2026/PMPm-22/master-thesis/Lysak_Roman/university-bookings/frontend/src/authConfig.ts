import { Configuration, PopupRequest } from '@azure/msal-browser';

export const msalConfig: Configuration = {
  auth: {
    clientId: import.meta.env.VITE_AZURE_CLIENT_ID as string,
    authority: `https://login.microsoftonline.com/${import.meta.env.VITE_AZURE_TENANT_ID}`,
    redirectUri: window.location.origin,
    postLogoutRedirectUri: window.location.origin,
  },
  cache: {
    cacheLocation: 'localStorage',
    storeAuthStateInCookie: false,
  },
};

// Scopes for the backend API (expose an API scope in Azure App Registration)
export const loginRequest: PopupRequest = {
  scopes: [
    'openid',
    'profile',
    'email',
    `api://${import.meta.env.VITE_AZURE_CLIENT_ID}/access_as_user`,
  ],
};

// Graph scopes (for reading user info on frontend only — profile picture etc.)
export const graphRequest = {
  scopes: ['User.Read'],
};
