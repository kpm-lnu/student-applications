import React, { useEffect, useState } from 'react';
import { makeStyles, Text, Spinner, Badge } from '@fluentui/react-components';
import {
  CalendarRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  ClockRegular,
} from '@fluentui/react-icons';
import api from '../../services/api';
import { DashboardStats } from '../../types';

const useStyles = makeStyles({
  root: {},
  pageTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 2.5vw, 28px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.02em',
    display: 'block',
    marginBottom: '28px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))',
    gap: '16px',
    marginBottom: '32px',
  },
  statCard: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    padding: '22px 24px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    borderTop: '4px solid var(--color-primary)',
  },
  iconRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    color: 'var(--color-secondary-3)',
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
  },
  statValue: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '2.2rem',
    color: 'var(--color-primary)',
    lineHeight: '1',
    display: 'block',
  },
  section: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    padding: '24px',
  },
  sectionTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '16px',
    color: 'var(--color-primary)',
    display: 'block',
    marginBottom: '16px',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
  },
  th: {
    textAlign: 'left' as const,
    padding: '10px 14px',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '700',
    letterSpacing: '0.04em',
    textTransform: 'uppercase' as const,
  },
  td: {
    padding: '11px 14px',
    borderBottom: '1px solid rgba(20,27,77,0.07)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    color: 'var(--color-text)',
  },
});

export function Dashboard() {
  const styles = useStyles();
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    api
      .get<DashboardStats>('/api/admin/stats')
      .then((r) => setStats(r.data))
      .catch(console.error)
      .finally(() => setIsLoading(false));
  }, []);

  if (isLoading) return <Spinner label="Завантаження аналітичної панелі..." />;
  if (!stats) return <Text style={{ fontFamily: 'var(--font-brand)' }}>Не вдалося завантажити статистику.</Text>;

  return (
    <div className={styles.root}>
      <span className={styles.pageTitle}>Аналітична панель</span>

      <div className={styles.grid}>
        <div className={styles.statCard}>
          <div className={styles.iconRow}>
            <CalendarRegular fontSize={18} />
            Записів сьогодні
          </div>
          <span className={styles.statValue}>{stats.totalBookingsToday}</span>
        </div>

        <div className={styles.statCard} style={{ borderTopColor: 'var(--color-secondary-2)' }}>
          <div className={styles.iconRow}>
            <CalendarRegular fontSize={18} />
            За місяць
          </div>
          <span className={styles.statValue}>{stats.totalBookingsThisMonth}</span>
        </div>

        <div className={styles.statCard} style={{ borderTopColor: 'var(--color-secondary-3)' }}>
          <div className={styles.iconRow}>
            <ClockRegular fontSize={18} />
            Очікують
          </div>
          <span className={styles.statValue}>{stats.pendingAppointments}</span>
        </div>

        <div className={styles.statCard} style={{ borderTopColor: '#9e9e9e' }}>
          <div className={styles.iconRow}>
            <DismissCircleRegular fontSize={18} />
            Скасувань
          </div>
          <span className={styles.statValue}>{stats.cancellationRate.toFixed(1)}%</span>
        </div>
      </div>

      <div className={styles.section}>
        <span className={styles.sectionTitle}>Популярні аудиторії</span>
        <table className={styles.table}>
          <thead>
            <tr>
              <th className={styles.th}>Приміщення</th>
              <th className={styles.th}>Кількість бронювань</th>
            </tr>
          </thead>
          <tbody>
            {stats.popularRooms.map((r) => (
              <tr key={r.roomId}>
                <td className={styles.td}>{r.roomName}</td>
                <td className={styles.td}>
                  <Badge
                    appearance="filled"
                    style={{ backgroundColor: 'var(--color-secondary-3)', color: 'var(--color-white)' }}
                  >
                    {r.count}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
