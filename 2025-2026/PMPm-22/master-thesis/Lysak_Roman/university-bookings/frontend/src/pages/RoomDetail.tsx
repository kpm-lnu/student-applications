import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  makeStyles,
  tokens,
  Text,
  Button,
  Spinner,
  Badge,
  Field,
  Textarea,
  Select,
} from '@fluentui/react-components';
import {
  CalendarRegular,
  ClockRegular,
  PersonRegular,
  PeopleRegular,
  ArrowLeftRegular,
  BuildingRegular,
  SportRegular,
  LockClosedRegular,
} from '@fluentui/react-icons';
import { Room, TimeSlot, DURATION_OPTIONS, DurationMinutes, Availability, SlotMode, UNIVERSITY_PARA } from '../types';
import { roomsService } from '../services/roomsService';
import { appointmentsService } from '../services/appointmentsService';
import { slotHoldsService } from '../services/slotHoldsService';
import { useAuth } from '../contexts/AuthContext';
import { useAppointments } from '../contexts/AppointmentsContext';
import { useNotifications } from '../contexts/NotificationContext';
import { useSignalR } from '../hooks/useSignalR';

const useStyles = makeStyles({
  root: { maxWidth: '800px', margin: '0 auto' },
  backBtn: { marginBottom: '16px' },
  header: { marginBottom: '24px' },
  typeRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: 'var(--color-secondary-3)',
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
    fontWeight: '700',
    marginBottom: '6px',
  },
  name: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 3vw, 30px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.01em',
    lineHeight: '1.2',
    display: 'block',
    marginBottom: '10px',
  },
  meta: {
    display: 'flex',
    gap: '16px',
    flexWrap: 'wrap',
    marginTop: '8px',
  },
  metaItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    color: tokens.colorNeutralForeground2,
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
  },
  description: {
    marginTop: '10px',
    color: tokens.colorNeutralForeground2,
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    lineHeight: '1.5',
    display: 'block',
  },
  section: {
    marginTop: '28px',
    padding: '24px',
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    borderLeft: '4px solid var(--color-primary)',
    '@media (max-width: 640px)': {
      padding: '16px 14px',
    },
  },
  sectionTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '15px',
    color: 'var(--color-primary)',
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    marginBottom: '16px',
  },
  dateGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(7, 1fr)',
    gap: '6px',
    marginBottom: '24px',
    '@media (max-width: 640px)': {
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '4px',
    },
  },
  slotsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(100px, 1fr))',
    gap: '8px',
    marginTop: '12px',
  },
});

function getNextDays(n = 14): Date[] {
  return Array.from({ length: n }, (_, i) => {
    const d = new Date();
    d.setDate(d.getDate() + i);
    return d;
  });
}

function toDateString(d: Date) {
  return d.toISOString().split('T')[0];
}

function isDayAvailable(d: Date, availability: Availability[]): boolean {
  return availability.some((a) => a.dayOfWeek === d.getDay());
}

function formatCountdown(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function getKyivHHMM(isoString: string): string {
  const parts = new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Europe/Kiev',
  }).formatToParts(new Date(isoString));
  const h = parts.find((p) => p.type === 'hour')?.value ?? '00';
  const m = parts.find((p) => p.type === 'minute')?.value ?? '00';
  return `${h}:${m}`;
}

function getParaForSlot(slot: TimeSlot) {
  const hhmm = getKyivHHMM(slot.startTime);
  return UNIVERSITY_PARA.find((p) => p.start === hhmm);
}

export function RoomDetail() {
  const styles = useStyles();
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { user, login } = useAuth();
  const { refresh } = useAppointments();
  const { addNotification } = useNotifications();
  const { on, off } = useSignalR('/hubs/appointments');

  const [room, setRoom] = useState<Room | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [duration, setDuration] = useState<DurationMinutes>(60);
  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [slots, setSlots] = useState<TimeSlot[]>([]);
  const [slotsLoading, setSlotsLoading] = useState(false);
  const [selectedSlot, setSelectedSlot] = useState<TimeSlot | null>(null);
  const [notes, setNotes] = useState('');
  const [booking, setBooking] = useState(false);
  const [error, setError] = useState('');

  // Slot hold state
  const [holdId, setHoldId] = useState<string | null>(null);
  const [holdExpiresAt, setHoldExpiresAt] = useState<Date | null>(null);
  const [holdSecondsLeft, setHoldSecondsLeft] = useState(0);
  const holdIdRef = useRef<string | null>(null);

  // Load room
  useEffect(() => {
    if (!id) return;
    roomsService
      .getById(id)
      .then((r) => {
        setRoom(r);
        const days = getNextDays(14);
        const first = days.find((d) => isDayAvailable(d, r.availability));
        if (first) setSelectedDate(first);
      })
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, [id]);

  // Load slots — release hold when date/duration changes
  useEffect(() => {
    if (!id || !room) return;

    const currentHoldId = holdIdRef.current;
    if (currentHoldId) {
      slotHoldsService.release(currentHoldId).catch(() => {});
      holdIdRef.current = null;
      setHoldId(null);
      setHoldExpiresAt(null);
    }

    setSlotsLoading(true);
    setSelectedSlot(null);
    const fetchDuration = room.slotMode === SlotMode.Para ? 80 : duration;
    roomsService
      .getAvailableSlots(id, toDateString(selectedDate), fetchDuration)
      .then(setSlots)
      .catch(console.error)
      .finally(() => setSlotsLoading(false));
  }, [id, selectedDate, duration, room]);

  // Release hold on SPA unmount or page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      const hId = holdIdRef.current;
      if (hId) slotHoldsService.releaseSync(hId);
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      const hId = holdIdRef.current;
      if (hId) slotHoldsService.release(hId).catch(() => {});
    };
  }, []);

  // Countdown timer
  useEffect(() => {
    if (!holdExpiresAt) {
      setHoldSecondsLeft(0);
      return;
    }

    const tick = () => {
      const left = Math.max(0, Math.floor((holdExpiresAt.getTime() - Date.now()) / 1000));
      setHoldSecondsLeft(left);

      if (left === 0) {
        holdIdRef.current = null;
        setHoldId(null);
        setHoldExpiresAt(null);
        setSelectedSlot(null);
        setError('Час тимчасового бронювання вийшов. Будь ласка, оберіть слот знову.');
      }
    };

    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, [holdExpiresAt]);

  // SignalR: listen for hold events from other users
  useEffect(() => {
    on('SlotHoldCreated', (data: unknown) => {
      const ev = data as { roomId: string; startDateTime: string };
      if (ev.roomId !== id) return;
      const evTime = new Date(ev.startDateTime).getTime();
      setSlots((prev) =>
        prev.map((s) =>
          new Date(s.startTime).getTime() === evTime
            ? { ...s, available: false, isHeld: true }
            : s
        )
      );
    });

    on('SlotHoldReleased', (data: unknown) => {
      const ev = data as { roomId: string; startDateTime: string };
      if (ev.roomId !== id) return;
      const evTime = new Date(ev.startDateTime).getTime();
      setSlots((prev) =>
        prev.map((s) =>
          new Date(s.startTime).getTime() === evTime
            ? { ...s, available: true, isHeld: false }
            : s
        )
      );
    });

    return () => {
      off('SlotHoldCreated');
      off('SlotHoldReleased');
    };
  }, [on, off, id]);

  const handleSelectSlot = async (slot: TimeSlot) => {
    if (selectedSlot?.startTime === slot.startTime) return;
    setError('');

    if (!user) {
      setSelectedSlot(slot);
      return;
    }

    // Release previous hold
    const currentHoldId = holdIdRef.current;
    if (currentHoldId) {
      slotHoldsService.release(currentHoldId).catch(() => {});
      holdIdRef.current = null;
      setHoldId(null);
      setHoldExpiresAt(null);
    }

    setSelectedSlot(slot);

    try {
      const hold = await slotHoldsService.create(id!, slot.startTime, slot.endTime);
      holdIdRef.current = hold.id;
      setHoldId(hold.id);
      setHoldExpiresAt(new Date(hold.expiresAt));
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message;
      setError(msg ?? 'Не вдалося зарезервувати час.');
      setSelectedSlot(null);
    }
  };

  const handleBook = async () => {
    if (!user) { login(); return; }
    if (!selectedSlot || !id) return;
    setBooking(true);
    setError('');
    try {
      const bookDuration: DurationMinutes = room?.slotMode === SlotMode.Para ? 80 : duration;
      await appointmentsService.create({
        roomId: id,
        durationMinutes: bookDuration,
        startDateTime: selectedSlot.startTime,
        notes: notes || undefined,
      });

      holdIdRef.current = null;
      setHoldId(null);
      setHoldExpiresAt(null);

      await refresh();
      addNotification({
        type: 'confirmed',
        title: 'Бронювання створено',
        body: `Запит на бронювання "${room?.name}" успішно надіслано. Очікуйте підтвердження на email.`,
      });
      navigate('/my-appointments');
    } catch (err: unknown) {
      const msg = (err as { response?: { data?: { message?: string } } })?.response?.data?.message;
      setError(msg ?? 'Помилка при бронюванні. Спробуйте ще раз.');
    } finally {
      setBooking(false);
    }
  };

  if (isLoading) return <Spinner label="Завантаження..." />;
  if (!room) return <Text style={{ fontFamily: 'var(--font-brand)' }}>Приміщення не знайдено.</Text>;

  const days = getNextDays(14);
  const TypeIcon = room.roomType?.name === 'sport' ? SportRegular : BuildingRegular;
  const visibleSlots = slots.filter((s) => s.available || s.isHeld);

  return (
    <div className={styles.root}>
      <Button
        className={styles.backBtn}
        appearance="subtle"
        icon={<ArrowLeftRegular />}
        onClick={() => navigate(-1)}
      >
        Назад
      </Button>

      <div className={styles.header}>
        <div className={styles.typeRow}>
          <TypeIcon fontSize={15} />
          {room.roomType?.label ?? '—'}
          {room.roomNumber && (
            <Badge appearance="outline" style={{ marginLeft: '4px', fontFamily: 'var(--font-brand)', fontSize: '11px' }}>
              № {room.roomNumber}
            </Badge>
          )}
        </div>

        <span className={styles.name}>{room.name}</span>

        <div className={styles.meta}>
          {room.capacity && (
            <span className={styles.metaItem}>
              <PeopleRegular fontSize={15} /> до {room.capacity} осіб
            </span>
          )}
          {room.responsiblePerson && (
            <span className={styles.metaItem}>
              <PersonRegular fontSize={15} />
              {room.responsiblePerson.displayName}
            </span>
          )}
        </div>

        {room.description && (
          <span className={styles.description}>{room.description}</span>
        )}
      </div>

      <div className={styles.section}>
          {room.slotMode === SlotMode.Interval && (
            <>
              <div className={styles.sectionTitle}>
                <ClockRegular fontSize={16} />
                Тривалість бронювання
              </div>
              <Field>
                <Select
                  value={String(duration)}
                  onChange={(_, d) => setDuration(Number(d.value) as DurationMinutes)}
                  style={{ fontFamily: 'var(--font-brand)', maxWidth: '320px' }}
                >
                  {DURATION_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </Select>
              </Field>
            </>
          )}

          <div className={styles.sectionTitle} style={{ marginTop: room.slotMode === SlotMode.Para ? '0' : '24px' }}>
            <CalendarRegular fontSize={16} />
            Оберіть дату
          </div>
          <div className={styles.dateGrid}>
            {days.map((d) => {
              const isSelected = toDateString(d) === toDateString(selectedDate);
              const available = isDayAvailable(d, room.availability);
              return (
                <Button
                  key={toDateString(d)}
                  appearance={isSelected ? 'primary' : 'outline'}
                  onClick={() => setSelectedDate(d)}
                  disabled={!available}
                  style={{ flexDirection: 'column', height: 'auto', padding: '8px 4px' }}
                >
                  <Text size={100}>
                    {d.toLocaleDateString('uk-UA', { weekday: 'short' })}
                  </Text>
                  <Text size={300} weight="semibold">
                    {d.getDate()}
                  </Text>
                </Button>
              );
            })}
          </div>

          <div className={styles.sectionTitle}>
            <ClockRegular fontSize={16} />
            Оберіть час
          </div>
          {slotsLoading ? (
            <Spinner size="small" label="Завантаження слотів..." style={{ marginTop: '8px' }} />
          ) : visibleSlots.length === 0 ? (
            <Text style={{ color: tokens.colorNeutralForeground3, display: 'block', marginTop: '8px', fontFamily: 'var(--font-brand)' }}>
              На обрану дату вільних слотів немає. Оберіть іншу дату або тривалість.
            </Text>
          ) : (
            <div className={styles.slotsGrid}
              style={room.slotMode === SlotMode.Para ? { gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))' } : undefined}
            >
              {visibleSlots.map((slot) => {
                const isSelected = selectedSlot?.startTime === slot.startTime;
                const para = room.slotMode === SlotMode.Para ? getParaForSlot(slot) : undefined;
                const time = para
                  ? `${para.label} ${para.start}–${para.end}`
                  : new Date(slot.startTime).toLocaleTimeString('uk-UA', {
                      hour: '2-digit',
                      minute: '2-digit',
                    });
                return (
                  <Button
                    key={slot.startTime}
                    appearance={isSelected ? 'primary' : 'outline'}
                    disabled={slot.isHeld && !isSelected}
                    onClick={() => slot.available ? handleSelectSlot(slot) : undefined}
                    style={{
                      fontFamily: 'var(--font-brand)',
                      flexDirection: 'column',
                      height: 'auto',
                      padding: '8px 6px',
                      ...(slot.isHeld && !isSelected ? { opacity: 0.55 } : {}),
                    }}
                  >
                    {slot.isHeld && !isSelected
                      ? <LockClosedRegular fontSize={11} style={{ color: 'var(--color-secondary-3)' }} />
                      : null}
                    <span>{time}</span>
                    {slot.isHeld && !isSelected && (
                      <span style={{ fontSize: '10px', color: 'var(--color-secondary-3)', fontWeight: '400' }}>
                        тимчасово
                      </span>
                    )}
                  </Button>
                );
              })}
            </div>
          )}

          {selectedSlot && (
            <>
              {holdId && holdSecondsLeft > 0 && (
                <div style={{
                  marginTop: '14px',
                  padding: '10px 14px',
                  background: 'rgba(72, 92, 199, 0.07)',
                  borderRadius: 'var(--radius-sm)',
                  fontFamily: 'var(--font-brand)',
                  fontSize: '13px',
                  color: holdSecondsLeft < 60 ? '#b45309' : 'var(--color-secondary-2)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                }}>
                  <ClockRegular fontSize={14} />
                  Слот тимчасово заброньовано — залишилось {formatCountdown(holdSecondsLeft)}
                </div>
              )}

              <Field label="Примітка (необов'язково)" style={{ marginTop: '20px' }}>
                <Textarea
                  value={notes}
                  onChange={(_, d) => setNotes(d.value)}
                  placeholder="Вкажіть деталі запиту..."
                  rows={3}
                  style={{ fontFamily: 'var(--font-brand)' }}
                />
              </Field>

              {error && (
                <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(183,28,28,0.07)', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-brand)', fontSize: '13px', color: '#b71c1c' }}>
                  {error}
                </div>
              )}

              <Button
                appearance="primary"
                onClick={handleBook}
                disabled={booking}
                style={{ marginTop: '16px', fontFamily: 'var(--font-brand)' }}
              >
                {booking ? 'Бронювання...' : user ? 'Підтвердити бронювання' : 'Увійти для бронювання'}
              </Button>
            </>
          )}

          {!selectedSlot && error && (
            <div style={{ marginTop: '12px', padding: '10px 14px', background: 'rgba(183,28,28,0.07)', borderRadius: 'var(--radius-sm)', fontFamily: 'var(--font-brand)', fontSize: '13px', color: '#b71c1c' }}>
              {error}
            </div>
          )}
        </div>
    </div>
  );
}
