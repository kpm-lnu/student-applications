import React, { useState, useRef, useCallback } from 'react';
import {
  ServiceBellRegular,
  ServiceBellFilled,
  CalendarAddRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
} from '@fluentui/react-icons';
import { Button, Text } from '@fluentui/react-components';
import { useNotifications, NotifType, AppNotification } from '../../contexts/NotificationContext';

const TYPE_ICON: Record<NotifType, React.ReactNode> = {
  new_booking: <CalendarAddRegular fontSize={15} style={{ color: 'var(--color-secondary-3)', flexShrink: 0 }} />,
  confirmed:   <CheckmarkCircleRegular fontSize={15} style={{ color: 'var(--color-secondary-2)', flexShrink: 0 }} />,
  cancelled:   <DismissCircleRegular fontSize={15} style={{ color: 'var(--color-primary)', flexShrink: 0 }} />,
};

function fmtTime(iso: string) {
  return new Date(iso).toLocaleString('uk-UA', {
    day: 'numeric', month: 'short',
    hour: '2-digit', minute: '2-digit',
    timeZone: 'Europe/Kyiv',
  });
}

function NotificationRow({ n }: { n: AppNotification }) {
  return (
    <div style={{
      padding: '10px 14px',
      borderBottom: '1px solid rgba(20,27,77,0.06)',
      backgroundColor: n.read ? 'transparent' : 'rgba(72,92,199,0.05)',
      display: 'flex', gap: 8, alignItems: 'flex-start',
    }}>
      <div style={{ marginTop: 1 }}>{TYPE_ICON[n.type]}</div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ fontFamily: 'var(--font-brand)', fontSize: 13, fontWeight: 600, color: 'var(--color-primary)', marginBottom: 2 }}>
          {n.title}
        </div>
        <div style={{ fontFamily: 'var(--font-brand)', fontSize: 12, color: 'rgba(20,27,77,0.60)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {n.body}
        </div>
      </div>
      <span style={{ fontFamily: 'var(--font-brand)', fontSize: 11, color: 'rgba(20,27,77,0.38)', flexShrink: 0, marginTop: 1 }}>
        {fmtTime(n.timestamp)}
      </span>
    </div>
  );
}

export function NotificationBell() {
  const { notifications, unreadCount, markAllRead, clearAll } = useNotifications();
  const [open, setOpen] = useState(false);
  const [dropTop, setDropTop] = useState(0);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout>>();

  const recent = notifications.slice(0, 15);

  const handleEnter = useCallback(() => {
    clearTimeout(closeTimer.current);
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      setDropTop(rect.top);
    }
    setOpen(true);
    if (unreadCount > 0) markAllRead();
  }, [unreadCount, markAllRead]);

  const handleLeave = useCallback(() => {
    closeTimer.current = setTimeout(() => setOpen(false), 150);
  }, []);

  const cancelLeave = useCallback(() => {
    clearTimeout(closeTimer.current);
  }, []);

  return (
    <>
      <button
        ref={triggerRef}
        className="notif-bell-btn"
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
      >
        <span style={{ position: 'relative', display: 'flex', alignItems: 'center', flexShrink: 0 }}>
          {unreadCount > 0 ? <ServiceBellFilled fontSize={20} /> : <ServiceBellRegular fontSize={20} />}
          {unreadCount > 0 && (
            <span style={{
              position: 'absolute', top: -6, right: -8,
              backgroundColor: 'var(--color-secondary-3)',
              color: '#fff', borderRadius: 8,
              minWidth: 15, height: 15,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 9, fontWeight: 700, lineHeight: 1, padding: '0 3px',
            }}>
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </span>
        Сповіщення
      </button>

      {open && (
        <div
          style={{
            position: 'fixed',
            left: 264,
            top: dropTop,
            width: 320,
            maxHeight: 420,
            zIndex: 9000,
            backgroundColor: '#fff',
            borderRadius: 'var(--radius-md)',
            boxShadow: '0 8px 32px rgba(20,27,77,0.18)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden',
          }}
          onMouseEnter={cancelLeave}
          onMouseLeave={handleLeave}
        >
          <div style={{
            padding: '12px 14px',
            borderBottom: '1px solid rgba(20,27,77,0.1)',
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            flexShrink: 0,
          }}>
            <Text weight="semibold" style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)', fontSize: 14 }}>
              Сповіщення
            </Text>
            {notifications.length > 0 && (
              <Button
                appearance="subtle" size="small" onClick={clearAll}
                style={{ fontFamily: 'var(--font-brand)', fontSize: 12, color: 'rgba(20,27,77,0.5)' }}
              >
                Очистити все
              </Button>
            )}
          </div>

          <div style={{ overflowY: 'auto', flex: 1 }}>
            {recent.length === 0 ? (
              <div style={{ padding: '28px 16px', textAlign: 'center', fontFamily: 'var(--font-brand)', fontSize: 13, color: 'rgba(20,27,77,0.40)' }}>
                Немає сповіщень
              </div>
            ) : (
              recent.map(n => <NotificationRow key={n.id} n={n} />)
            )}
          </div>
        </div>
      )}
    </>
  );
}
