import React from 'react';
import {
  makeStyles,
  tokens,
  Button,
  Text,
  Card,
} from '@fluentui/react-components';
import { PersonRegular } from '@fluentui/react-icons';
import { useAuth } from '../contexts/AuthContext';
import { Navigate } from 'react-router-dom';

const useStyles = makeStyles({
  root: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: tokens.colorNeutralBackground2,
  },
  card: {
    width: '380px',
    padding: '40px',
    textAlign: 'center',
  },
  logo: {
    marginBottom: '32px',
  },
  actions: {
    marginTop: '24px',
  },
});

export function LoginPage() {
  const styles = useStyles();
  const { user, login, isLoading } = useAuth();

  if (user) return <Navigate to="/" replace />;

  return (
    <div className={styles.root}>
      <Card className={styles.card}>
        <div className={styles.logo}>
          <Text size={800} weight="bold" style={{ color: tokens.colorBrandForeground1 }} block>
            UniBook
          </Text>
          <Text size={400} style={{ color: tokens.colorNeutralForeground3 }}>
            Система бронювання університету
          </Text>
        </div>

        <Text size={300} block style={{ color: tokens.colorNeutralForeground2, marginBottom: '24px' }}>
          Увійдіть через університетський акаунт Microsoft для доступу до системи бронювання.
        </Text>

        <Button
          appearance="primary"
          size="large"
          icon={<PersonRegular />}
          onClick={login}
          disabled={isLoading}
          style={{ width: '100%' }}
        >
          Увійти через Microsoft
        </Button>
      </Card>
    </div>
  );
}
