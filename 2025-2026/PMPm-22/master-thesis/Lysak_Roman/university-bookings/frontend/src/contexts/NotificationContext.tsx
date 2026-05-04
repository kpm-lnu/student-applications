import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';

export type NotifType = 'new_booking' | 'confirmed' | 'cancelled';

export interface AppNotification {
  id: string;
  type: NotifType;
  title: string;
  body: string;
  timestamp: string;
  read: boolean;
  appointmentId?: string;
}

export interface ToastItem extends Omit<AppNotification, 'read'> {
  leaving: boolean;
}

interface NotificationContextValue {
  notifications: AppNotification[];
  toasts: ToastItem[];
  unreadCount: number;
  addNotification: (n: Omit<AppNotification, 'id' | 'timestamp' | 'read'>) => void;
  dismissToast: (id: string) => void;
  markAllRead: () => void;
  clearAll: () => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

const STORAGE_KEY = 'uni_notifications_v1';
const MAX_STORED = 50;
const TOAST_MS = 5000;
const EXIT_MS = 320;

export function NotificationProvider({ children }: { children: React.ReactNode }) {
  const [notifications, setNotifications] = useState<AppNotification[]>(() => {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]'); }
    catch { return []; }
  });
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(notifications.slice(0, MAX_STORED))); }
    catch { /* ignore quota errors */ }
  }, [notifications]);

  const startExit = useCallback((id: string) => {
    setToasts(prev => prev.map(t => t.id === id ? { ...t, leaving: true } : t));
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), EXIT_MS);
  }, []);

  const addNotification = useCallback((n: Omit<AppNotification, 'id' | 'timestamp' | 'read'>) => {
    const id = crypto.randomUUID();
    const timestamp = new Date().toISOString();

    setNotifications(prev => [{ ...n, id, timestamp, read: false }, ...prev].slice(0, MAX_STORED));
    setToasts(prev => [...prev, { ...n, id, timestamp, leaving: false }]);

    const autoExit = setTimeout(() => startExit(id), TOAST_MS);
    return () => clearTimeout(autoExit);
  }, [startExit]);

  const dismissToast = useCallback((id: string) => startExit(id), [startExit]);

  const markAllRead = useCallback(() => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  }, []);

  const clearAll = useCallback(() => {
    setNotifications([]);
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* ignore */ }
  }, []);

  return (
    <NotificationContext.Provider value={{
      notifications,
      toasts,
      unreadCount: notifications.filter(n => !n.read).length,
      addNotification,
      dismissToast,
      markAllRead,
      clearAll,
    }}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const ctx = useContext(NotificationContext);
  if (!ctx) throw new Error('useNotifications must be inside NotificationProvider');
  return ctx;
}
