import React, { useEffect, useState, useMemo } from 'react';
import { makeStyles, Spinner, Badge, Select, Input } from '@fluentui/react-components';
import { SearchRegular } from '@fluentui/react-icons';
import { Appointment, AppointmentStatus, DURATION_OPTIONS } from '../../types';
import { appointmentsService } from '../../services/appointmentsService';
import { Pagination } from '../../components/shared/Pagination';

const PAGE_SIZE = 10;

const useStyles = makeStyles({
  root: {},
  pageTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 2.5vw, 28px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.02em',
    display: 'block',
    marginBottom: '20px',
  },
  topBar: {
    display: 'flex',
    gap: '12px',
    marginBottom: '20px',
    flexWrap: 'wrap',
    alignItems: 'center',
  },
  searchWrap: {
    flex: '1 1 240px',
    maxWidth: '340px',
  },
  tableWrap: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    overflow: 'auto',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse' as const,
  },
  th: {
    textAlign: 'left' as const,
    padding: '11px 14px',
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
    verticalAlign: 'middle' as const,
  },
  actions: {
    display: 'flex',
    gap: '6px',
    flexWrap: 'wrap',
  },
  confirmBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '700',
    background: 'var(--color-primary)',
    color: 'var(--color-white)',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 10px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  },
  cancelBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '600',
    background: 'transparent',
    color: '#b71c1c',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 10px',
    cursor: 'pointer',
  },
  empty: {
    padding: '32px',
    textAlign: 'center' as const,
    fontFamily: 'var(--font-brand)',
    color: 'rgba(20,27,77,0.45)',
    fontSize: '14px',
  },
});

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

export function StaffAppointmentsPage() {
  const styles = useStyles();
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  const load = () => {
    setIsLoading(true);
    appointmentsService
      .getStaff(statusFilter ? { status: statusFilter } : undefined)
      .then(setAppointments)
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };

  useEffect(() => { load(); }, [statusFilter]);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return appointments;
    return appointments.filter(
      (a) =>
        (a.clientUser?.displayName ?? '').toLowerCase().includes(q) ||
        (a.roomName ?? '').toLowerCase().includes(q),
    );
  }, [appointments, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const handleSearch = (val: string) => { setSearchQuery(val); setCurrentPage(1); };
  const handleStatusFilter = (val: string) => { setStatusFilter(val); setSearchQuery(''); setCurrentPage(1); };

  const updateStatus = async (id: string, status: AppointmentStatus) => {
    try {
      await appointmentsService.updateStaffStatus(id, { status });
      setAppointments((prev) => prev.map((a) => (a.id === id ? { ...a, status } : a)));
    } catch {
      alert('Не вдалося оновити статус.');
    }
  };

  return (
    <div className={styles.root}>
      <span className={styles.pageTitle}>Записи моїх приміщень</span>

      <div className={styles.topBar}>
        <div className={styles.searchWrap}>
          <Input
            placeholder="Пошук за клієнтом або приміщенням..."
            value={searchQuery}
            onChange={(_, d) => handleSearch(d.value)}
            contentBefore={<SearchRegular fontSize={16} />}
            style={{ fontFamily: 'var(--font-brand)', width: '100%' }}
          />
        </div>
        <Select
          value={statusFilter}
          onChange={(_, d) => handleStatusFilter(d.value)}
          style={{ fontFamily: 'var(--font-brand)' }}
        >
          <option value="">Всі статуси</option>
          <option value="Pending">Очікують</option>
          <option value="Confirmed">Підтверджені</option>
          <option value="Cancelled">Скасовані</option>
          <option value="Completed">Завершені</option>
        </Select>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження..." />
      ) : (
        <>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Клієнт</th>
                  <th className={styles.th}>Приміщення</th>
                  <th className={styles.th}>Тривалість</th>
                  <th className={styles.th}>Дата та час</th>
                  <th className={styles.th}>Статус</th>
                  <th className={styles.th}>Дії</th>
                </tr>
              </thead>
              <tbody>
                {paginated.length === 0 ? (
                  <tr>
                    <td colSpan={6} className={styles.empty}>Нічого не знайдено</td>
                  </tr>
                ) : (
                  paginated.map((a) => (
                    <tr key={a.id}>
                      <td className={styles.td}>{a.clientUser?.displayName ?? '—'}</td>
                      <td className={styles.td} style={{ fontWeight: '600', color: 'var(--color-primary)' }}>
                        {a.roomName}
                      </td>
                      <td className={styles.td}>
                        {DURATION_OPTIONS.find((o) => o.value === a.durationMinutes)?.label ?? `${a.durationMinutes} хв`}
                      </td>
                      <td className={styles.td}>
                        {new Date(a.startDateTime).toLocaleString('uk-UA', {
                          dateStyle: 'short',
                          timeStyle: 'short',
                        })}
                      </td>
                      <td className={styles.td}>
                        <Badge color={statusColor[a.status]} appearance="filled">
                          {statusLabel[a.status]}
                        </Badge>
                      </td>
                      <td className={styles.td}>
                        <div className={styles.actions}>
                          {a.status === AppointmentStatus.Pending && (
                            <button
                              className={styles.confirmBtn}
                              onClick={() => updateStatus(a.id, AppointmentStatus.Confirmed)}
                              onMouseOver={(e) => (e.currentTarget.style.background = 'var(--color-secondary-2)')}
                              onMouseOut={(e) => (e.currentTarget.style.background = 'var(--color-primary)')}
                            >
                              Підтвердити
                            </button>
                          )}
                          {(a.status === AppointmentStatus.Pending || a.status === AppointmentStatus.Confirmed) && (
                            <button
                              className={styles.cancelBtn}
                              onClick={() => updateStatus(a.id, AppointmentStatus.Cancelled)}
                            >
                              Скасувати
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
          <Pagination
            currentPage={currentPage}
            totalPages={totalPages}
            totalItems={filtered.length}
            pageSize={PAGE_SIZE}
            onPageChange={setCurrentPage}
          />
        </>
      )}
    </div>
  );
}
