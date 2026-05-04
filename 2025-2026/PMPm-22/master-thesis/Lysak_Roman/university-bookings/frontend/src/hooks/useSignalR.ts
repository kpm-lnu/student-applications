import { useEffect, useRef, useCallback } from 'react';
import * as signalR from '@microsoft/signalr';
import { msalInstance } from '../main';
import { loginRequest } from '../authConfig';

type EventHandler = (data: unknown) => void;

export function useSignalR(hubUrl: string) {
  const connectionRef = useRef<signalR.HubConnection | null>(null);
  const handlersRef = useRef<Map<string, EventHandler>>(new Map());

  useEffect(() => {
    if (!hubUrl) return;

    let cancelled = false;

    const connection = new signalR.HubConnectionBuilder()
      .withUrl(hubUrl, {
        accessTokenFactory: async () => {
          const accounts = msalInstance.getAllAccounts();
          if (accounts.length === 0) return '';
          const result = await msalInstance.acquireTokenSilent({
            ...loginRequest,
            account: accounts[0],
          });
          return result.accessToken;
        },
      })
      .withAutomaticReconnect()
      .configureLogging(signalR.LogLevel.Warning)
      .build();

    connectionRef.current = connection;

    // Re-register all handlers on this new connection instance
    // (covers React StrictMode double-invoke and hubUrl changes)
    handlersRef.current.forEach((handler, event) => {
      connection.on(event, handler);
    });

    connection.onreconnected(() => {
      handlersRef.current.forEach((handler, event) => {
        connection.off(event);
        connection.on(event, handler);
      });
    });

    connection.start().catch((err) => {
      if (!cancelled) console.error('SignalR connection error', err);
    });

    return () => {
      cancelled = true;
      connection.stop();
    };
  }, [hubUrl]);

  const on = useCallback((event: string, handler: EventHandler) => {
    handlersRef.current.set(event, handler);
    if (connectionRef.current) {
      connectionRef.current.off(event);
      connectionRef.current.on(event, handler);
    }
  }, []);

  const off = useCallback((event: string) => {
    handlersRef.current.delete(event);
    connectionRef.current?.off(event);
  }, []);

  const invoke = useCallback(async (method: string, ...args: unknown[]) => {
    return connectionRef.current?.invoke(method, ...args);
  }, []);

  return { on, off, invoke };
}
