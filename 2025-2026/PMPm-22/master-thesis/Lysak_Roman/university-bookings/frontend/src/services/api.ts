import axios from 'axios';
import { msalInstance } from '../main';
import { loginRequest } from '../authConfig';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL ?? 'http://localhost:5000',
  headers: { 'Content-Type': 'application/json' },
});

export let cachedAccessToken: string | null = null;

// Attach Bearer token to every request
api.interceptors.request.use(async (config) => {
  const accounts = msalInstance.getAllAccounts();
  if (accounts.length > 0) {
    try {
      const result = await msalInstance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      cachedAccessToken = result.accessToken;
      config.headers.Authorization = `Bearer ${result.accessToken}`;
    } catch {
      // Silent refresh failed — force interactive login
      await msalInstance.acquireTokenRedirect(loginRequest);
    }
  }
  return config;
});

export default api;
