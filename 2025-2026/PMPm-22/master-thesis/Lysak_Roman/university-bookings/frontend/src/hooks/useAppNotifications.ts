import { useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { useNotifications } from '../contexts/NotificationContext';
import { useSignalR } from './useSignalR';
import { Appointment, UserRole } from '../types';

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString('uk-UA', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Europe/Kyiv',
  });
}

export function useAppNotifications() {
  const { user } = useAuth();
  const { addNotification } = useNotifications();
  const { on } = useSignalR(user ? '/hubs/appointments' : '');

  useEffect(() => {
    if (!user) return;

    if (user.role === UserRole.Admin || user.role === UserRole.Staff) {
      on('NewAppointmentCreated', (raw) => {
        const a = raw as Appointment;
        addNotification({
          type: 'new_booking',
          title: 'Нове бронювання',
          body: `${a.roomName} · ${a.clientUser?.displayName ?? ''} · ${fmtDate(a.startDateTime)}`,
          appointmentId: a.id,
        });
      });
    }

    if (user.role === UserRole.Staff) {
      on('AppointmentCancelled', (raw) => {
        const a = raw as Appointment;
        addNotification({
          type: 'cancelled',
          title: 'Бронювання скасовано клієнтом',
          body: `${a.roomName} · ${a.clientUser?.displayName ?? ''} · ${fmtDate(a.startDateTime)}`,
          appointmentId: a.id,
        });
      });
    }

    if (user.role === UserRole.Student) {
      on('AppointmentConfirmed', (raw) => {
        const a = raw as Appointment;
        addNotification({
          type: 'confirmed',
          title: 'Бронювання підтверджено',
          body: `${a.roomName} · ${fmtDate(a.startDateTime)}`,
          appointmentId: a.id,
        });
      });

      on('AppointmentCancelled', (raw) => {
        const a = raw as Appointment;
        addNotification({
          type: 'cancelled',
          title: 'Бронювання скасовано',
          body: a.cancellationReason
            ? `${a.roomName} · Причина: ${a.cancellationReason}`
            : `${a.roomName} · ${fmtDate(a.startDateTime)}`,
          appointmentId: a.id,
        });
      });
    }
  }, [user, on, addNotification]);
}
