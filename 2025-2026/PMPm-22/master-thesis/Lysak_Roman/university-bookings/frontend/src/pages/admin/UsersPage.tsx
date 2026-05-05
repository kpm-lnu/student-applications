import React, { useEffect, useState, useMemo } from 'react';
import {
  makeStyles,
  Spinner,
  Select,
  Avatar,
  Input,
  Dialog,
  DialogSurface,
  DialogBody,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogTrigger,
  Field,
} from '@fluentui/react-components';
import { SearchRegular, DeleteRegular, PeopleRegular } from '@fluentui/react-icons';
import { User, UserRole } from '../../types';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import { Pagination } from '../../components/shared/Pagination';

const PAGE_SIZE = 10;

// Default cutoff: start of previous academic year (Sep 1, 2025)
const DEFAULT_BULK_DATE = '2025-09-01';

const useStyles = makeStyles({
  root: {},
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '20px',
    flexWrap: 'wrap',
    gap: '12px',
  },
  pageTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: 'clamp(22px, 2.5vw, 28px)',
    color: 'var(--color-primary)',
    letterSpacing: '0.02em',
  },
  bulkBtn: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '13px',
    backgroundColor: 'transparent',
    color: '#b71c1c',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-md)',
    padding: '8px 16px',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    transition: 'background 0.2s, color 0.2s',
  },
  topBar: {
    marginBottom: '20px',
  },
  searchWrap: {
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
    padding: '11px 16px',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '700',
    letterSpacing: '0.04em',
    textTransform: 'uppercase' as const,
  },
  td: {
    padding: '12px 16px',
    borderBottom: '1px solid rgba(20,27,77,0.07)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    color: 'var(--color-text)',
  },
  userCell: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  userName: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    color: 'var(--color-primary)',
  },
  selfBadge: {
    display: 'inline-block',
    fontSize: '11px',
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    color: 'var(--color-primary)',
    background: 'rgba(20,27,77,0.08)',
    borderRadius: 'var(--radius-sm)',
    padding: '2px 7px',
    marginLeft: '7px',
    verticalAlign: 'middle',
    letterSpacing: '0.03em',
  },
  deleteBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    fontWeight: '600',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-sm)',
    padding: '4px 10px',
    cursor: 'pointer',
    background: 'transparent',
    color: '#b71c1c',
    display: 'inline-flex',
    alignItems: 'center',
    gap: '4px',
    transition: 'background 0.15s',
  },
  empty: {
    padding: '32px',
    textAlign: 'center' as const,
    fontFamily: 'var(--font-brand)',
    color: 'rgba(20,27,77,0.45)',
    fontSize: '14px',
  },
  bulkDialogText: {
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    color: 'var(--color-text)',
    lineHeight: '1.6',
    marginBottom: '16px',
    display: 'block',
  },
  bulkDateRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    flexWrap: 'wrap',
  },
  warningBox: {
    background: '#fff3f3',
    border: '1.5px solid #e57373',
    borderRadius: 'var(--radius-sm)',
    padding: '10px 14px',
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    color: '#b71c1c',
    marginTop: '14px',
    lineHeight: '1.5',
  },
  confirmDeleteBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    fontWeight: '700',
    background: '#b71c1c',
    color: 'var(--color-white)',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    padding: '7px 18px',
    cursor: 'pointer',
    transition: 'background 0.2s',
  },
});

export function UsersPage() {
  const styles = useStyles();
  const { user: currentUser } = useAuth();
  const [users, setUsers] = useState<User[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const [bulkDialogOpen, setBulkDialogOpen] = useState(false);
  const [bulkDate, setBulkDate] = useState(DEFAULT_BULK_DATE);
  const [isBulkDeleting, setIsBulkDeleting] = useState(false);

  const loadUsers = () => {
    api
      .get<User[]>('/api/admin/users')
      .then((r) => setUsers(r.data))
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };

  useEffect(() => { loadUsers(); }, []);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return users;
    return users.filter(
      (u) =>
        u.displayName.toLowerCase().includes(q) ||
        u.email.toLowerCase().includes(q),
    );
  }, [users, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const handleSearch = (val: string) => {
    setSearchQuery(val);
    setCurrentPage(1);
  };

  const handleRoleChange = async (userId: string, role: UserRole) => {
    try {
      await api.put(`/api/admin/users/${userId}/role`, { role });
      setUsers((prev) =>
        prev.map((u) => (u.id === userId ? { ...u, role } : u)),
      );
    } catch (err: unknown) {
      const message =
        (err as { response?: { data?: unknown } })?.response?.data;
      alert(typeof message === 'string' ? message : 'Не вдалося змінити роль.');
    }
  };

  const handleDeleteUser = async (userId: string, displayName: string) => {
    if (!confirm(`Видалити користувача "${displayName}"? Усі його записи будуть також видалені.`)) return;
    try {
      await api.delete(`/api/admin/users/${userId}`);
      setUsers((prev) => prev.filter((u) => u.id !== userId));
    } catch {
      alert('Не вдалося видалити користувача.');
    }
  };

  const handleBulkDelete = async () => {
    if (!bulkDate) return;
    setIsBulkDeleting(true);
    try {
      const res = await api.delete<{ deleted: number }>(
        `/api/admin/users/bulk-students?before=${bulkDate}`,
      );
      const count = res.data.deleted;
      setBulkDialogOpen(false);
      loadUsers();
      alert(`Видалено ${count} студент${count === 1 ? 'а' : 'ів'}.`);
    } catch {
      alert('Помилка масового видалення.');
    } finally {
      setIsBulkDeleting(false);
    }
  };

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <span className={styles.pageTitle}>Користувачі ({users.length})</span>
        <button
          className={styles.bulkBtn}
          onClick={() => setBulkDialogOpen(true)}
          onMouseOver={(e) => {
            e.currentTarget.style.background = '#fff3f3';
          }}
          onMouseOut={(e) => {
            e.currentTarget.style.background = 'transparent';
          }}
        >
          <PeopleRegular fontSize={16} />
          Очищення неактивних студентів
        </button>
      </div>

      <div className={styles.topBar}>
        <div className={styles.searchWrap}>
          <Input
            placeholder="Пошук за іменем або email..."
            value={searchQuery}
            onChange={(_, d) => handleSearch(d.value)}
            contentBefore={<SearchRegular fontSize={16} />}
            style={{ fontFamily: 'var(--font-brand)', width: '100%' }}
          />
        </div>
      </div>

      {isLoading ? (
        <Spinner label="Завантаження..." />
      ) : (
        <>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.th}>Користувач</th>
                  <th className={styles.th}>Email</th>
                  <th className={styles.th}>Роль</th>
                  <th className={styles.th}>Зареєстрований</th>
                  <th className={styles.th}>Дії</th>
                </tr>
              </thead>
              <tbody>
                {paginated.length === 0 ? (
                  <tr>
                    <td colSpan={5} className={styles.empty}>Нічого не знайдено</td>
                  </tr>
                ) : (
                  paginated.map((u) => {
                    const isSelf = u.id === currentUser?.id;
                    return (
                      <tr key={u.id}>
                        <td className={styles.td}>
                          <div className={styles.userCell}>
                            <Avatar name={u.displayName} size={28} />
                            <span className={styles.userName}>{u.displayName}</span>
                            {isSelf && <span className={styles.selfBadge}>Це ви</span>}
                          </div>
                        </td>
                        <td className={styles.td}>{u.email}</td>
                        <td className={styles.td}>
                          <Select
                            value={u.role}
                            disabled={isSelf}
                            onChange={(_, d) => handleRoleChange(u.id, d.value as UserRole)}
                            style={{ minWidth: '120px', fontFamily: 'var(--font-brand)', opacity: isSelf ? 0.45 : 1 }}
                          >
                            <option value={UserRole.Student}>Student</option>
                            <option value={UserRole.Staff}>Staff</option>
                            <option value={UserRole.Admin}>Admin</option>
                          </Select>
                        </td>
                        <td className={styles.td}>
                          {new Date(u.createdAt).toLocaleDateString('uk-UA')}
                        </td>
                        <td className={styles.td}>
                          <button
                            className={styles.deleteBtn}
                            disabled={isSelf}
                            onClick={() => handleDeleteUser(u.id, u.displayName)}
                            onMouseOver={(e) => { if (!isSelf) e.currentTarget.style.background = '#fff3f3'; }}
                            onMouseOut={(e) => (e.currentTarget.style.background = 'transparent')}
                            style={{ opacity: isSelf ? 0.35 : 1, cursor: isSelf ? 'not-allowed' : 'pointer' }}
                          >
                            <DeleteRegular fontSize={13} /> Видалити
                          </button>
                        </td>
                      </tr>
                    );
                  })
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

      {/* Bulk delete dialog */}
      <Dialog open={bulkDialogOpen} onOpenChange={(_, d) => setBulkDialogOpen(d.open)}>
        <DialogSurface>
          <DialogBody>
            <DialogTitle
              style={{ fontFamily: 'var(--font-brand)', color: 'var(--color-primary)', fontWeight: '700' }}
            >
              Очищення неактивних студентів
            </DialogTitle>
            <DialogContent>
              <span className={styles.bulkDialogText}>
                Видаліть усіх студентів, які вже не навчаються в університеті, зареєстрованих до:
              </span>
              <div className={styles.bulkDateRow}>
                <Field label="Дата включно">
                  <Input
                    type="date"
                    value={bulkDate}
                    onChange={(_, d) => setBulkDate(d.value)}
                    style={{ fontFamily: 'var(--font-brand)' }}
                  />
                </Field>
              </div>
              <div className={styles.warningBox}>
                Увага: буде видалено всіх користувачів з роллю <strong>Student</strong>, зареєстрованих до{' '}
                <strong>{bulkDate ? new Date(bulkDate).toLocaleDateString('uk-UA') : '—'}</strong> включно.
                Усі їхні записи на бронювання також будуть видалені. Цю дію неможливо скасувати.
              </div>
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
                    padding: '7px 16px',
                    cursor: 'pointer',
                    color: 'var(--color-primary)',
                  }}
                >
                  Скасувати
                </button>
              </DialogTrigger>
              <button
                className={styles.confirmDeleteBtn}
                onClick={handleBulkDelete}
                disabled={!bulkDate || isBulkDeleting}
                onMouseOver={(e) => (e.currentTarget.style.background = '#7f0000')}
                onMouseOut={(e) => (e.currentTarget.style.background = '#b71c1c')}
              >
                {isBulkDeleting ? 'Видалення...' : 'Підтвердити видалення'}
              </button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>
    </div>
  );
}
