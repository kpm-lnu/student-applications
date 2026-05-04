import React, {
  createContext,
  useContext,
  useEffect,
  useState,
  useCallback,
} from 'react';
import { useMsal, useIsAuthenticated } from '@azure/msal-react';
import { InteractionStatus } from '@azure/msal-browser';
import { loginRequest } from '../authConfig';
import { User } from '../types';
import api from '../services/api';

interface AuthContextValue {
  user: User | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: () => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  isLoading: true,
  isAuthenticated: false,
  login: () => {},
  logout: () => {},
});

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const { instance, accounts, inProgress } = useMsal();
  const isAuthenticated = useIsAuthenticated();
  const [user, setUser] = useState<User | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const syncUser = useCallback(async () => {
    // Wait until MSAL has finished all in-progress interactions (redirect handling etc.)
    if (inProgress !== InteractionStatus.None) return;

    if (!isAuthenticated || accounts.length === 0) {
      setUser(null);
      setIsLoading(false);
      return;
    }
    try {
      // POST /api/auth/login — backend validates JWT and upserts the user
      const res = await api.post<User>('/api/auth/login');
      setUser(res.data);
    } catch (err) {
      console.error('User sync failed', err);
      setUser(null);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated, accounts, inProgress]);

  useEffect(() => {
    syncUser();
  }, [syncUser]);

  const login = () =>
    instance.loginRedirect(loginRequest).catch(console.error);

  const logout = () =>
    instance
      .logoutRedirect({ postLogoutRedirectUri: window.location.origin })
      .catch(console.error);

  return (
    <AuthContext.Provider
      value={{ user, isLoading, isAuthenticated, login, logout }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
