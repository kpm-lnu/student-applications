import React, { useState } from 'react';
import { Link, Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  makeStyles,
  tokens,
  Button,
  Avatar,
  Text,
  Tooltip,
} from '@fluentui/react-components';
import {
  CalendarRegular,
  HomeRegular,
  SignOutRegular,
  SettingsRegular,
  NavigationRegular,
} from '@fluentui/react-icons';
import { useAuth } from '../../contexts/AuthContext';
import { UserRole } from '../../types';
import logoShort from '../../assets/logo/logo-ua-short.svg';
import { NotificationBell } from '../Notifications/NotificationBell';
import { ToastContainer } from '../Notifications/ToastContainer';
import { useAppNotifications } from '../../hooks/useAppNotifications';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    height: '100vh',
    backgroundColor: 'var(--color-bg)',
    overflow: 'hidden',
  },
  // Hidden on desktop, shown on mobile via .app-layout-mobile-header CSS class
  mobileHeader: {
    display: 'none',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '0 16px',
    height: '56px',
    backgroundColor: 'var(--color-primary)',
    boxShadow: 'var(--shadow-nav)',
    position: 'fixed',
    top: '0',
    left: '0',
    right: '0',
    zIndex: 100,
    flexShrink: '0',
  },
  mobileLogoImg: {
    height: '36px',
    width: 'auto',
    display: 'block',
  },
  overlay: {
    position: 'fixed',
    inset: '0',
    backgroundColor: 'rgba(0,0,0,0.5)',
    zIndex: 150,
    cursor: 'pointer',
  },
  sidebar: {
    width: '256px',
    backgroundColor: 'var(--color-primary)',
    display: 'flex',
    flexDirection: 'column',
    padding: '0 0 16px',
    gap: '2px',
    boxShadow: 'var(--shadow-nav)',
    flexShrink: '0',
  },
  logoBlock: {
    padding: '20px 20px 16px',
    borderBottom: '1px solid rgba(255,255,255,0.12)',
    marginBottom: '8px',
  },
  logoImg: {
    width: '100%',
    maxWidth: '200px',
    display: 'block',
  },
  navItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '11px 20px',
    borderRadius: '0',
    textDecoration: 'none',
    color: 'rgba(255,255,255,0.82)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    transition: 'background 0.15s, color 0.15s',
    ':hover': {
      backgroundColor: 'rgba(255,255,255,0.10)',
      color: 'var(--color-white)',
    },
  },
  navItemActive: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '11px 20px',
    borderRadius: '0',
    textDecoration: 'none',
    color: 'var(--color-white)',
    fontFamily: 'var(--font-brand)',
    fontSize: '14px',
    fontWeight: '700',
    backgroundColor: 'rgba(255,255,255,0.13)',
    borderLeft: '3px solid var(--color-white)',
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    padding: '32px',
  },
  userSection: {
    marginTop: 'auto',
    borderTop: '1px solid rgba(255,255,255,0.12)',
    paddingTop: '12px',
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    padding: '12px 8px 0',
  },
  userInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '8px 12px',
  },
});

export function AppLayout() {
  const styles = useStyles();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const [isMobileOpen, setIsMobileOpen] = useState(false);
  useAppNotifications();

  const closeMobileMenu = () => setIsMobileOpen(false);

  const navClass = (to: string, exact = false) =>
    (exact ? pathname === to : pathname.startsWith(to))
      ? styles.navItemActive
      : styles.navItem;

  return (
    <div className={styles.root}>
      {/* Mobile top bar — shown via CSS on mobile */}
      <div className={`${styles.mobileHeader} app-layout-mobile-header`}>
        <Link to="/" onClick={closeMobileMenu}>
          <img src={logoShort} alt="Львівський університет" className={styles.mobileLogoImg} />
        </Link>
        <Button
          appearance="subtle"
          icon={<NavigationRegular fontSize={24} />}
          onClick={() => setIsMobileOpen(true)}
          style={{ color: 'var(--color-white)', minWidth: 0 }}
          aria-label="Відкрити меню"
        />
      </div>

      {/* Overlay — only rendered when mobile menu is open */}
      {isMobileOpen && (
        <div className={styles.overlay} onClick={closeMobileMenu} />
      )}

      {/* Sidebar — slides in on mobile via CSS */}
      <aside
        className={`${styles.sidebar} app-layout-sidebar${isMobileOpen ? ' app-layout-sidebar--open' : ''}`}
      >
        <div className={styles.logoBlock}>
          <Link to="/" onClick={closeMobileMenu}>
            <img src={logoShort} alt="Львівський університет" className={styles.logoImg} />
          </Link>
        </div>

        <Link to="/" className={navClass('/', true)} onClick={closeMobileMenu}>
          <HomeRegular fontSize={20} />
          Приміщення
        </Link>

        {user && (
          <Link to="/my-appointments" className={navClass('/my-appointments')} onClick={closeMobileMenu}>
            <CalendarRegular fontSize={20} />
            Мої записи
          </Link>
        )}

        {user?.role === UserRole.Staff && (
          <Link to="/staff/appointments" className={navClass('/staff/appointments')} onClick={closeMobileMenu}>
            <CalendarRegular fontSize={20} />
            Записи приміщень
          </Link>
        )}

        {user?.role === UserRole.Admin && (
          <Link to="/admin" className={navClass('/admin')} onClick={closeMobileMenu}>
            <SettingsRegular fontSize={20} />
            Адмін-панель
          </Link>
        )}

        {user && <NotificationBell />}

        <div className={styles.userSection}>
          {user ? (
            <>
              <div className={styles.userInfo}>
                <Avatar name={user.displayName} size={32} />
                <div>
                  <Text
                    size={200}
                    weight="semibold"
                    block
                    style={{ color: 'var(--color-white)', fontFamily: 'var(--font-brand)' }}
                  >
                    {user.displayName}
                  </Text>
                  <Text
                    size={100}
                    block
                    style={{ color: 'rgba(255,255,255,0.55)', fontFamily: 'var(--font-brand)' }}
                  >
                    {user.role}
                  </Text>
                </div>
              </div>
              <Tooltip content="Вийти" relationship="label">
                <Button
                  appearance="subtle"
                  icon={<SignOutRegular />}
                  onClick={logout}
                  style={{
                    justifyContent: 'flex-start',
                    paddingLeft: '12px',
                    color: 'rgba(255,255,255,0.75)',
                    fontFamily: 'var(--font-brand)',
                  }}
                >
                  Вийти
                </Button>
              </Tooltip>
            </>
          ) : (
            <Button
              appearance="primary"
              onClick={() => { navigate('/login'); closeMobileMenu(); }}
              style={{
                margin: '0 12px',
                backgroundColor: 'var(--color-secondary-2)',
                fontFamily: 'var(--font-brand)',
              }}
            >
              Увійти
            </Button>
          )}
        </div>
      </aside>

      <main className={`${styles.content} app-layout-content`}>
        <Outlet />
      </main>

      <ToastContainer />
    </div>
  );
}
