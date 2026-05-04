import React, { useState } from 'react';
import {
  Badge,
  Text,
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogActions,
  DialogContent,
  Field,
  Textarea,
  makeStyles,
  tokens,
} from '@fluentui/react-components';
import { CalendarRegular, ClockRegular } from '@fluentui/react-icons';
import { Appointment, AppointmentStatus, DURATION_OPTIONS } from '../../types';
import { appointmentsService } from '../../services/appointmentsService';

const statusColor: Record<AppointmentStatus, 'informative' | 'success' | 'danger' | 'subtle'> = {
  [AppointmentStatus.Pending]: 'informative',
  [AppointmentStatus.Confirmed]: 'success',
  [AppointmentStatus.Cancelled]: 'danger',
  [AppointmentStatus.Completed]: 'subtle',
};

const statusLabel: Record<AppointmentStatus, string> = {
  [AppointmentStatus.Pending]: 'Очікує',
  [AppointmentStatus.Confirmed]: 'Підтверджено',
  [AppointmentStatus.Cancelled]: 'Скасовано',
  [AppointmentStatus.Completed]: 'Завершено',
};

const statusBorderColor: Record<AppointmentStatus, string> = {
  [AppointmentStatus.Pending]:   '#485CC7',
  [AppointmentStatus.Confirmed]: '#1E22AA',
  [AppointmentStatus.Cancelled]: '#9e9e9e',
  [AppointmentStatus.Completed]: '#c0c0c0',
};

const useStyles = makeStyles({
  card: {
    width: '100%',
    background: 'var(--color-white)',
    boxShadow: 'var(--shadow-card)',
    borderRadius: 'var(--radius-md)',
    padding: '20px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '12px',
    borderLeft: '4px solid var(--color-primary)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    gap: '12px',
  },
  roomName: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '16px',
    color: 'var(--color-primary)',
    letterSpacing: '0.01em',
    lineHeight: '1.3',
  },
  meta: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  metaRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    color: tokens.colorNeutralForeground2,
    fontSize: '13px',
    fontFamily: 'var(--font-brand)',
  },
  notes: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    color: tokens.colorNeutralForeground3,
    fontStyle: 'italic',
    paddingTop: '2px',
  },
  footer: {
    paddingTop: '4px',
    borderTop: '1px solid rgba(20,27,77,0.07)',
  },
  cancelBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    fontWeight: '600',
    color: '#b71c1c',
    background: 'transparent',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-md)',
    padding: '6px 16px',
    cursor: 'pointer',
    transition: 'background 0.15s, color 0.15s',
  },
});

interface Props {
  appointment: Appointment;
  onCancelled?: () => void;
}

export function AppointmentCard({ appointment, onCancelled }: Props) {
  const styles = useStyles();
  const [reason, setReason] = useState('');
  const [isCancelling, setIsCancelling] = useState(false);

  const canCancel =
    appointment.status === AppointmentStatus.Pending ||
    appointment.status === AppointmentStatus.Confirmed;

  const handleCancel = async () => {
    setIsCancelling(true);
    try {
      await appointmentsService.cancel(appointment.id, reason || undefined);
      onCancelled?.();
    } catch (err) {
      console.error('Cancel failed', err);
    } finally {
      setIsCancelling(false);
    }
  };

  const startDate = new Date(appointment.startDateTime);
  const borderColor = statusBorderColor[appointment.status];

  return (
    <div className={styles.card} style={{ borderLeftColor: borderColor }}>
      <div className={styles.header}>
        <span className={styles.roomName}>
          {appointment.roomName}
        </span>
        <Badge color={statusColor[appointment.status]} appearance="filled">
          {statusLabel[appointment.status]}
        </Badge>
      </div>

      <div className={styles.meta}>
        <div className={styles.metaRow}>
          <CalendarRegular fontSize={15} />
          {startDate.toLocaleDateString('uk-UA', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
          })}
        </div>
        <div className={styles.metaRow}>
          <ClockRegular fontSize={15} />
          {startDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' })}
          {' — '}
          {new Date(appointment.endDateTime).toLocaleTimeString('uk-UA', {
            hour: '2-digit',
            minute: '2-digit',
          })}
          {' · '}
          {DURATION_OPTIONS.find((o) => o.value === appointment.durationMinutes)?.label ?? `${appointment.durationMinutes} хв`}
        </div>
        {appointment.notes && (
          <span className={styles.notes}>Примітка: {appointment.notes}</span>
        )}
      </div>

      {canCancel && (
        <div className={styles.footer}>
          <Dialog>
            <DialogTrigger disableButtonEnhancement>
              <button
                className={styles.cancelBtn}
                onMouseOver={(e) => {
                  e.currentTarget.style.background = '#fdecea';
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
              >
                Скасувати запис
              </button>
            </DialogTrigger>
            <DialogSurface>
              <DialogBody>
                <DialogTitle style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)' }}>
                  Скасувати запис?
                </DialogTitle>
                <DialogContent>
                  <Text style={{ fontFamily: 'var(--font-brand)' }}>
                    Ви впевнені, що хочете скасувати бронювання{' '}
                    <strong>{appointment.roomName}</strong>?
                  </Text>
                  <Field label="Причина скасування (необов'язково)" style={{ marginTop: '16px' }}>
                    <Textarea
                      value={reason}
                      onChange={(_, d) => setReason(d.value)}
                      placeholder="Вкажіть причину..."
                      style={{ fontFamily: 'var(--font-brand)' }}
                    />
                  </Field>
                </DialogContent>
                <DialogActions>
                  <DialogTrigger disableButtonEnhancement>
                    <button
                      style={{
                        fontFamily: 'var(--font-brand)',
                        fontSize: '13px',
                        background: 'transparent',
                        border: '1.5px solid rgba(20,27,77,0.25)',
                        borderRadius: 'var(--radius-md)',
                        padding: '6px 16px',
                        cursor: 'pointer',
                        color: 'var(--color-primary)',
                      }}
                    >
                      Ні, залишити
                    </button>
                  </DialogTrigger>
                  <button
                    style={{
                      fontFamily: 'var(--font-brand)',
                      fontSize: '13px',
                      fontWeight: '700',
                      background: '#c62828',
                      color: '#fff',
                      border: 'none',
                      borderRadius: 'var(--radius-md)',
                      padding: '6px 16px',
                      cursor: isCancelling ? 'not-allowed' : 'pointer',
                      opacity: isCancelling ? 0.65 : 1,
                    }}
                    onClick={handleCancel}
                    disabled={isCancelling}
                  >
                    {isCancelling ? 'Скасування...' : 'Так, скасувати'}
                  </button>
                </DialogActions>
              </DialogBody>
            </DialogSurface>
          </Dialog>
        </div>
      )}
    </div>
  );
}
