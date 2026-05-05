import React, { useEffect, useState, useMemo } from 'react';
import {
  makeStyles,
  Spinner,
  Avatar,
  Input,
} from '@fluentui/react-components';
import { SearchRegular } from '@fluentui/react-icons';
import { StaffMember } from '../../types';
import api from '../../services/api';
import { Pagination } from '../../components/shared/Pagination';

const PAGE_SIZE = 9;

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
    marginBottom: '20px',
  },
  searchWrap: {
    maxWidth: '340px',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    gap: '18px',
    '@media (min-width: 900px)': {
      gridTemplateColumns: 'repeat(2, 1fr)',
    },
    '@media (min-width: 1280px)': {
      gridTemplateColumns: 'repeat(3, 1fr)',
    },
  },
  card: {
    background: 'var(--color-white)',
    borderRadius: 'var(--radius-md)',
    boxShadow: 'var(--shadow-card)',
    padding: '22px',
    borderLeft: '4px solid var(--color-secondary-2)',
    display: 'flex',
    flexDirection: 'column',
    gap: '0',
  },
  cardHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '14px',
    marginBottom: '12px',
  },
  name: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '15px',
    color: 'var(--color-primary)',
    display: 'block',
  },
  email: {
    fontFamily: 'var(--font-brand)',
    fontSize: '12px',
    color: 'var(--color-secondary-3)',
    display: 'block',
    marginTop: '2px',
  },
  empty: {
    fontFamily: 'var(--font-brand)',
    color: 'rgba(20,27,77,0.45)',
    fontSize: '14px',
    padding: '32px 0',
    textAlign: 'center' as const,
    gridColumn: '1 / -1',
  },
});

export function StaffPage() {
  const styles = useStyles();
  const [staff, setStaff] = useState<StaffMember[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);

  const load = () => {
    api.get<StaffMember[]>('/api/admin/staff')
      .then((r) => setStaff(r.data))
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };
  useEffect(() => { load(); }, []);

  const filtered = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    if (!q) return staff;
    return staff.filter((s) => s.displayName.toLowerCase().includes(q) || s.email.toLowerCase().includes(q));
  }, [staff, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const paginated = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const handleSearch = (val: string) => {
    setSearchQuery(val);
    setCurrentPage(1);
  };

  return (
    <div className={styles.root}>
      <span className={styles.pageTitle}>Персонал</span>

      <div className={styles.topBar}>
        <div className={styles.searchWrap}>
          <Input
            placeholder="Пошук за іменем або поштою..."
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
          <div className={styles.grid}>
            {paginated.length === 0 ? (
              <div className={styles.empty}>Нічого не знайдено</div>
            ) : (
              paginated.map((s) => (
                <div key={s.id} className={styles.card}>
                  <div className={styles.cardHeader}>
                    <Avatar name={s.displayName} size={48} />
                    <div>
                      <span className={styles.name}>{s.displayName}</span>
                      <span className={styles.email}>{s.email}</span>
                    </div>
                  </div>
                </div>
              ))
            )}
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
