import React, { useMemo } from 'react';
import {
  makeStyles,
  tokens,
  Text,
  Spinner,
  Tab,
  TabList,
} from '@fluentui/react-components';
import { useAppointments } from '../contexts/AppointmentsContext';
import { AppointmentCard } from '../components/shared/AppointmentCard';
import { AppointmentStatus } from '../types';
import { useAuth } from '../contexts/AuthContext';

const useStyles = makeStyles({
  root: { maxWidth: '860px', margin: '0 auto' },
  header: { marginBottom: '28px' },
  title: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 3vw, 30px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.02em',
    display: 'block',
    marginBottom: '4px',
  },
  subtitle: {
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    color: 'var(--color-secondary-3)',
    display: 'block',
  },
  tabList: {
    borderBottom: '2px solid rgba(20, 27, 77, 0.10)',
    marginBottom: '4px',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
    marginTop: '20px',
  },
  empty: {
    textAlign: 'center',
    padding: '64px 0',
    color: tokens.colorNeutralForeground3,
    fontFamily: 'var(--font-brand)',
  },
  emptyIcon: {
    fontSize: '40px',
    display: 'block',
    marginBottom: '12px',
    opacity: '0.35',
  },
});

type TabValue = 'upcoming' | 'past';

export function MyAppointments() {
  const styles = useStyles();
  const { user } = useAuth();
  const { appointments, isLoading, refresh } = useAppointments();
  const [tab, setTab] = React.useState<TabValue>('upcoming');

  const now = new Date();

  const upcoming = useMemo(
    () =>
      appointments.filter(
        (a) =>
          (a.status === AppointmentStatus.Pending ||
            a.status === AppointmentStatus.Confirmed) &&
          new Date(a.startDateTime) >= now,
      ),
    [appointments],
  );

  const past = useMemo(
    () =>
      appointments.filter(
        (a) =>
          a.status === AppointmentStatus.Completed ||
          a.status === AppointmentStatus.Cancelled ||
          new Date(a.startDateTime) < now,
      ),
    [appointments],
  );

  if (!user) {
    return (
      <div className={styles.empty}>
        <Text size={400} style={{ fontFamily: 'var(--font-brand)' }}>
          Увійдіть, щоб переглядати свої записи.
        </Text>
      </div>
    );
  }

  const displayed = tab === 'upcoming' ? upcoming : past;

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.title}>Мої записи</span>
        <span className={styles.subtitle}>
          Управляйте своїми бронюваннями університетських приміщень
        </span>
      </div>

      <div className={styles.tabList}>
        <TabList
          selectedValue={tab}
          onTabSelect={(_, d) => setTab(d.value as TabValue)}
          style={{ fontFamily: 'var(--font-brand)' }}
        >
          <Tab value="upcoming" style={{ fontFamily: 'var(--font-brand)', fontWeight: tab === 'upcoming' ? '700' : '400' }}>
            Майбутні ({upcoming.length})
          </Tab>
          <Tab value="past" style={{ fontFamily: 'var(--font-brand)', fontWeight: tab === 'past' ? '700' : '400' }}>
            Минулі ({past.length})
          </Tab>
        </TabList>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження..." style={{ marginTop: '40px' }} />
      ) : displayed.length === 0 ? (
        <div className={styles.empty}>
          <span className={styles.emptyIcon}>📋</span>
          <Text size={400} style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)', opacity: 0.5 }}>
            {tab === 'upcoming'
              ? 'У вас немає майбутніх записів.'
              : 'Минулих записів немає.'}
          </Text>
        </div>
      ) : (
        <div className={styles.list}>
          {displayed.map((a) => (
            <AppointmentCard key={a.id} appointment={a} onCancelled={refresh} />
          ))}
        </div>
      )}
    </div>
  );
}
