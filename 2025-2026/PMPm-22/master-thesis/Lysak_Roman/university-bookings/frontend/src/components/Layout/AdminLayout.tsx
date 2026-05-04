import React from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { makeStyles } from '@fluentui/react-components';
import {
  DataBarVerticalRegular,
  PeopleRegular,
  ServiceBellRegular,
  CalendarRegular,
  PersonAccountsRegular,
  TagRegular,
  LocationRegular,
  ArrowLeftRegular,
} from '@fluentui/react-icons';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    height: '100%',
  },
  sidebar: {
    width: '210px',
    backgroundColor: 'var(--color-secondary-1)',
    display: 'flex',
    flexDirection: 'column',
    padding: '0 0 16px',
    gap: '2px',
    flexShrink: '0',
  },
  sidebarHeader: {
    padding: '18px 20px 14px',
    borderBottom: '1px solid rgba(255,255,255,0.12)',
    marginBottom: '8px',
  },
  sidebarTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '13px',
    color: 'rgba(255,255,255,0.55)',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    display: 'block',
  },
  sidebarSub: {
    fontFamily: 'var(--font-brand)',
    fontSize: '15px',
    fontWeight: '700',
    color: 'var(--color-white)',
    display: 'block',
    marginTop: '2px',
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '9px',
    padding: '10px 20px',
    textDecoration: 'none',
    color: 'rgba(255,255,255,0.75)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    transition: 'background 0.15s, color 0.15s',
    whiteSpace: 'nowrap',
    ':hover': {
      backgroundColor: 'rgba(255,255,255,0.10)',
      color: 'var(--color-white)',
    },
  },
  navItemActive: {
    display: 'flex',
    alignItems: 'center',
    gap: '9px',
    padding: '10px 20px',
    textDecoration: 'none',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    fontWeight: '700',
    backgroundColor: 'rgba(255,255,255,0.13)',
    borderLeft: '3px solid var(--color-white)',
    whiteSpace: 'nowrap',
  },
  backLink: {
    display: 'flex',
    alignItems: 'center',
    gap: '9px',
    padding: '10px 20px',
    textDecoration: 'none',
    color: 'rgba(255,255,255,0.50)',
    fontFamily: 'var(--font-brand)',
    fontSize: '13px',
    transition: 'color 0.15s',
    marginTop: 'auto',
    whiteSpace: 'nowrap',
    ':hover': {
      color: 'var(--color-white)',
    },
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    padding: '32px',
    backgroundColor: 'var(--color-bg)',
  },
});

const navLinks = [
  { to: '/admin',              icon: <DataBarVerticalRegular fontSize={18} />, label: 'Аналітична панель', exact: true },
  { to: '/admin/users',        icon: <PeopleRegular fontSize={18} />,          label: 'Користувачі',      exact: false },
  { to: '/admin/rooms',        icon: <ServiceBellRegular fontSize={18} />,     label: 'Аудиторії та зали', exact: false },
  { to: '/admin/room-types',   icon: <TagRegular fontSize={18} />,             label: 'Типи приміщень',   exact: false },
  { to: '/admin/addresses',    icon: <LocationRegular fontSize={18} />,        label: 'Адреси',           exact: false },
  { to: '/admin/staff',        icon: <PersonAccountsRegular fontSize={18} />,  label: 'Персонал',         exact: false },
  { to: '/admin/appointments', icon: <CalendarRegular fontSize={18} />,        label: 'Записи',           exact: false },
];

export function AdminLayout() {
  const styles = useStyles();
  const { pathname } = useLocation();

  const isActive = (to: string, exact: boolean) =>
    exact ? pathname === to : pathname.startsWith(to);

  return (
    <div className={`${styles.root} admin-layout-root`}>
      <aside className={`${styles.sidebar} admin-layout-sidebar`}>
        <div className={`${styles.sidebarHeader} admin-layout-sidebar-header`}>
          <span className={styles.sidebarTitle}>Управління</span>
          <span className={styles.sidebarSub}>Адмін-панель</span>
        </div>

        {navLinks.map(({ to, icon, label, exact }) => (
          <Link
            key={to}
            to={to}
            className={
              isActive(to, exact)
                ? `${styles.navItemActive} admin-layout-nav-item-active`
                : `${styles.navItem} admin-layout-nav-item`
            }
          >
            {icon}
            {label}
          </Link>
        ))}

        <Link to="/" className={`${styles.backLink} admin-layout-back-link`}>
          <ArrowLeftRegular fontSize={16} />
          На головну
        </Link>
      </aside>

      <main className={`${styles.content} admin-layout-content`}>
        <Outlet />
      </main>
    </div>
  );
}
