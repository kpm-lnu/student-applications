import React, { useState } from 'react';
import { supabase } from '../supabaseClient';
import { theme } from '../theme';
import bgImage from '../assets/background.png'; // Імпортуємо твій фон
import { LogIn, UserPlus } from 'lucide-react';

const Auth = () => {
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isRegistering, setIsRegistering] = useState(false);

  const handleAuth = async (e) => {
    e.preventDefault();
    setLoading(true);
    
    const { error } = isRegistering 
      ? await supabase.auth.signUp({ email, password })
      : await supabase.auth.signInWithPassword({ email, password });

    if (error) alert(error.message);
    setLoading(false);
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center',
      backgroundImage: `url(${bgImage})`, // Твій фон з листочками
      ...theme.backgroundConfig,
    }}>
      <div style={{ 
        background: 'rgba(255, 255, 255, 0.9)', // Біла напівпрозора плашка
        padding: '40px', 
        borderRadius: '24px', 
        boxShadow: '0 10px 25px rgba(0,0,0,0.2)',
        width: '100%',
        maxWidth: '400px',
        textAlign: 'center',
        backdropFilter: 'blur(5px)'
      }}>
        <h2 style={{ color: '#1e293b', marginBottom: '30px' }}>
          {isRegistering ? 'Реєстрація' : 'Вхід у щоденник'}
        </h2>

        <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
          <input
            type="email"
            placeholder="Ваш Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ 
              padding: '12px', 
              borderRadius: '12px', 
              border: '1px solid #cbd5e1',
              outline: 'none'
            }}
            required
          />
          <input
            type="password"
            placeholder="Пароль"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ 
              padding: '12px', 
              borderRadius: '12px', 
              border: '1px solid #cbd5e1',
              outline: 'none'
            }}
            required
          />
          <button
            disabled={loading}
            style={{
              background: theme.colors.primary,
              color: theme.colors.primaryDark,
              padding: '14px',
              borderRadius: '12px',
              border: 'none',
              fontWeight: 'bold',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              marginTop: '10px'
            }}
          >
            {isRegistering ? <UserPlus size={20} /> : <LogIn size={20} />}
            {loading ? 'Зачекайте...' : (isRegistering ? 'Створити аккаунт' : 'Увійти')}
          </button>
        </form>

        <p 
          onClick={() => setIsRegistering(!isRegistering)}
          style={{ 
            marginTop: '20px', 
            color: '#64748b', 
            cursor: 'pointer', 
            fontSize: '14px',
            textDecoration: 'underline'
          }}
        >
          {isRegistering ? 'Вже є аккаунт? Увійдіть' : 'Немає аккаунту? Зареєструйтеся'}
        </p>
      </div>
    </div>
  );
};

export default Auth;