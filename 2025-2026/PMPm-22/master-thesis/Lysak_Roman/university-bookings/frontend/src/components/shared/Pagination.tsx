import React from 'react';
import { makeStyles } from '@fluentui/react-components';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '16px',
    marginBottom: '16px',
    paddingRight: '72px',
    flexWrap: 'wrap',
    gap: '10px',
  },
  info: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    color: 'rgba(20,27,77,0.6)',
  },
  controls: {
    display: 'flex',
    gap: '4px',
    alignItems: 'center',
  },
  pageBtn: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '32px',
    height: '32px',
    padding: '0 8px',
    border: '1.5px solid rgba(20,27,77,0.2)',
    borderRadius: 'var(--radius-sm)',
    background: 'transparent',
    color: 'var(--color-primary)',
    cursor: 'pointer',
    transition: 'background 0.15s, border-color 0.15s',
  },
  pageBtnActive: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    fontWeight: '700',
    minWidth: '32px',
    height: '32px',
    padding: '0 8px',
    border: '1.5px solid var(--color-primary)',
    borderRadius: 'var(--radius-sm)',
    background: 'var(--color-primary)',
    color: 'var(--color-white)',
    cursor: 'default',
  },
  pageBtnDisabled: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    fontWeight: '600',
    minWidth: '32px',
    height: '32px',
    padding: '0 8px',
    border: '1.5px solid rgba(20,27,77,0.1)',
    borderRadius: 'var(--radius-sm)',
    background: 'transparent',
    color: 'rgba(20,27,77,0.3)',
    cursor: 'not-allowed',
  },
  dots: {
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    color: 'rgba(20,27,77,0.4)',
    padding: '0 4px',
  },
});

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  totalItems: number;
  pageSize: number;
  onPageChange: (page: number) => void;
}

export function Pagination({ currentPage, totalPages, totalItems, pageSize, onPageChange }: PaginationProps) {
  const styles = useStyles();

  if (totalPages <= 1) return null;

  const from = (currentPage - 1) * pageSize + 1;
  const to = Math.min(currentPage * pageSize, totalItems);

  const pages: (number | '...')[] = [];
  if (totalPages <= 7) {
    for (let i = 1; i <= totalPages; i++) pages.push(i);
  } else {
    pages.push(1);
    if (currentPage > 3) pages.push('...');
    for (let i = Math.max(2, currentPage - 1); i <= Math.min(totalPages - 1, currentPage + 1); i++) {
      pages.push(i);
    }
    if (currentPage < totalPages - 2) pages.push('...');
    pages.push(totalPages);
  }

  return (
    <div className={styles.root}>
      <span className={styles.info}>
        {from}–{to} з {totalItems}
      </span>
      <div className={styles.controls}>
        <button
          className={currentPage === 1 ? styles.pageBtnDisabled : styles.pageBtn}
          disabled={currentPage === 1}
          onClick={() => onPageChange(currentPage - 1)}
          onMouseOver={(e) => { if (currentPage !== 1) e.currentTarget.style.background = 'rgba(20,27,77,0.06)'; }}
          onMouseOut={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          ←
        </button>
        {pages.map((p, i) =>
          p === '...' ? (
            <span key={`dots-${i}`} className={styles.dots}>…</span>
          ) : (
            <button
              key={p}
              className={p === currentPage ? styles.pageBtnActive : styles.pageBtn}
              onClick={() => p !== currentPage && onPageChange(p as number)}
              onMouseOver={(e) => { if (p !== currentPage) e.currentTarget.style.background = 'rgba(20,27,77,0.06)'; }}
              onMouseOut={(e) => { if (p !== currentPage) e.currentTarget.style.background = 'transparent'; }}
            >
              {p}
            </button>
          )
        )}
        <button
          className={currentPage === totalPages ? styles.pageBtnDisabled : styles.pageBtn}
          disabled={currentPage === totalPages}
          onClick={() => onPageChange(currentPage + 1)}
          onMouseOver={(e) => { if (currentPage !== totalPages) e.currentTarget.style.background = 'rgba(20,27,77,0.06)'; }}
          onMouseOut={(e) => { e.currentTarget.style.background = 'transparent'; }}
        >
          →
        </button>
      </div>
    </div>
  );
}
