import React from 'react';
import {
  CalendarAddRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  DismissRegular,
} from '@fluentui/react-icons';
import { useNotifications, NotifType } from '../../contexts/NotificationContext';

const CONFIG: Record<NotifType, { border: string; icon: React.ReactNode }> = {
  new_booking: {
    border: 'var(--color-secondary-3)',
    icon: <CalendarAddRegular fontSize={18} style={{ color: 'var(--color-secondary-3)', flexShrink: 0 }} />,
  },
  confirmed: {
    border: 'var(--color-secondary-2)',
    icon: <CheckmarkCircleRegular fontSize={18} style={{ color: 'var(--color-secondary-2)', flexShrink: 0 }} />,
  },
  cancelled: {
    border: 'var(--color-primary)',
    icon: <DismissCircleRegular fontSize={18} style={{ color: 'var(--color-primary)', flexShrink: 0 }} />,
  },
};

export function ToastContainer() {
  const { toasts, dismissToast } = useNotifications();
  if (toasts.length === 0) return null;

  return (
    <div style={{
      position: 'fixed',
      top: 16,
      right: 16,
      zIndex: 9999,
      display: 'flex',
      flexDirection: 'column',
      gap: 10,
      width: 340,
      pointerEvents: 'none',
    }}>
      {toasts.map(t => {
        const cfg = CONFIG[t.type];
        return (
          <div
            key={t.id}
            className={t.leaving ? 'toast-leaving' : 'toast-entering'}
            style={{
              display: 'flex',
              alignItems: 'flex-start',
              gap: 10,
              padding: '12px 14px',
              background: '#fff',
              borderLeft: `4px solid ${cfg.border}`,
              borderRadius: 'var(--radius-md)',
              boxShadow: 'var(--shadow-card)',
              pointerEvents: 'all',
            }}
          >
            <div style={{ marginTop: 1 }}>{cfg.icon}</div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{
                fontFamily: 'var(--font-brand)',
                fontWeight: 700,
                fontSize: 13,
                color: 'var(--color-primary)',
                marginBottom: 2,
              }}>
                {t.title}
              </div>
              <div style={{
                fontFamily: 'var(--font-brand)',
                fontSize: 12,
                color: 'rgba(20,27,77,0.60)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}>
                {t.body}
              </div>
            </div>
            <button
              onClick={() => dismissToast(t.id)}
              aria-label="Закрити"
              style={{
                background: 'none',
                border: 'none',
                padding: 0,
                cursor: 'pointer',
                color: 'rgba(20,27,77,0.35)',
                display: 'flex',
                alignItems: 'center',
                flexShrink: 0,
              }}
            >
              <DismissRegular fontSize={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
