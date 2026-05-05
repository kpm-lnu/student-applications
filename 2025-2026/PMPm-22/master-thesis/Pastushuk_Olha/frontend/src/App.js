import React, { useState, useEffect } from 'react';
import { supabase } from './supabaseClient';
import Auth from './components/Auth';
import EmotionRecorder from './EmotionRecorder';
import Dashboard from './components/Dashboard';
import Profile from './components/Profile'; 
import Statistics from './components/Statistics';
import { theme } from './theme';
import bgImage from './assets/background.png';

function App() {
  const [session, setSession] = useState(null);
  const [view, setView] = useState('recorder');

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => setSession(session));
    
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session);
    });

    return () => subscription.unsubscribe();
  }, []);

  const navButtonStyle = (isActive) => ({
    background: isActive ? theme.colors.primary : 'transparent',
    color: isActive ? theme.colors.primaryDark : 'white',
    border: `1px solid ${theme.colors.primary}`,
    padding: '8px 20px',
    borderRadius: '20px',
    cursor: 'pointer',
    fontWeight: isActive ? 'bold' : 'normal',
    transition: theme.transition
  });

  if (!session) return <Auth />;

  return (
    <div className="App" style={{ 
      minHeight: '100vh', 
      backgroundImage: `url(${bgImage})`, 
      ...theme.backgroundConfig, 
      color: 'white' 
    }}>
      {/* Навігаційна панель */}
      <nav style={{ 
        padding: '15px 30px', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between', 
        background: theme.colors.buttonCard,
        boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
        position: 'relative',
        zIndex: 10
      }}>
        
        {/* Назва сайту зліва */}
        <div style={{ 
          fontSize: '30px', 
          fontWeight: 'bold', 
          color: theme.colors.primary,
          letterSpacing: '0.5px',
          whiteSpace: 'nowrap'
        }}>
          Помічник для емоцій
        </div>

        {/* Контейнер кнопок по центру */}
        <div style={{ 
          display: 'flex', 
          gap: '12px', 
          position: 'absolute', 
          left: '50%', 
          transform: 'translateX(-50%)',
          flexWrap: 'nowrap'
        }}>
          <button 
            onClick={() => setView('recorder')} 
            style={navButtonStyle(view === 'recorder')}
          >
            Записати
          </button>
          
          <button 
            onClick={() => setView('dashboard')} 
            style={navButtonStyle(view === 'dashboard')}
          >
            Мої записи
          </button>

          <button 
            onClick={() => setView('statistics')} 
            style={navButtonStyle(view === 'statistics')}
          >
            Статистика
          </button>

          <button 
            onClick={() => setView('profile')} 
            style={navButtonStyle(view === 'profile')}
          >
            Профіль
          </button>
        </div>

        {/* Кнопка виходу справа */}
        <button 
          onClick={() => supabase.auth.signOut()} 
          style={{ 
            background: theme.colors.danger, 
            border: 'none', 
            color: 'white', 
            padding: '8px 20px', 
            borderRadius: '20px', 
            cursor: 'pointer',
            fontWeight: '500',
            transition: 'opacity 0.2s'
          }}
          onMouseOver={(e) => e.target.style.opacity = '0.9'}
          onMouseOut={(e) => e.target.style.opacity = '1'}
        >
          Вийти
        </button>
      </nav>

      {/* Контент сторінок */}
      <main style={{ padding: '40px 20px', position: 'relative' }}>
        {view === 'recorder' && <EmotionRecorder user={session.user} />}
        {view === 'dashboard' && <Dashboard user={session.user} />}
        {view === 'statistics' && <Statistics user={session.user} />}
        {view === 'profile' && <Profile user={session.user} />}
      </main>
    </div>
  );
}

export default App;