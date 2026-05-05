import React, { useState, useEffect } from 'react';
import { supabase } from '../supabaseClient';
import { Mail, BarChart, Save, Edit2, Check, X } from 'lucide-react';
import { theme } from '../theme';

const Profile = ({ user }) => {
    const [loading, setLoading] = useState(false);
    const [count, setCount] = useState(0);
    const [isEditing, setIsEditing] = useState(false);
    // Призначаємо ім'я за замовчуванням, якщо метадані порожні
    const [displayName, setDisplayName] = useState(user.user_metadata?.display_name || 'Користувач ' + user.id.slice(0, 4));

    useEffect(() => {
        getStats();
    }, []);

    const getStats = async () => {
        const { count, error } = await supabase
            .from('entries')
            .select('*', { count: 'exact', head: true });
        if (!error) setCount(count);
    };

    const handleUpdateName = async () => {
        setLoading(true);
        const { error } = await supabase.auth.updateUser({
            data: { display_name: displayName }
        });

        if (error) {
            alert("Помилка: " + error.message);
        } else {
            setIsEditing(false); // Повертаємося до режиму тексту
        }
        setLoading(false);
    };

    return (
        <div style={{ padding: '40px', maxWidth: '500px', margin: '0 auto', color: 'white' }}>
            
            {/* ВЕРХНЯ СЕКЦІЯ: ВІТАННЯ ТА РЕДАГУВАННЯ */}
            <div style={{ textAlign: 'center', marginBottom: '30px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '10px' }}>
                {isEditing ? (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                        <input 
                            type="text" 
                            value={displayName} 
                            onChange={(e) => setDisplayName(e.target.value)}
                            autoFocus
                            style={{ 
                                fontSize: '24px', fontWeight: 'bold', background: theme.colors.background, 
                                border: '1px solid #4ade80', color: 'white', padding: '5px 15px', 
                                borderRadius: '10px', textAlign: 'center' 
                            }}
                        />
                        <button onClick={handleUpdateName} style={{ background: theme.colors.primary, border: 'none', padding: '8px', borderRadius: '50%', cursor: 'pointer' }}>
                            <Check size={20} color="white" />
                        </button>
                        <button onClick={() => setIsEditing(false)} style={{ background: theme.colors.card, border: 'none', padding: '8px', borderRadius: '50%', cursor: 'pointer' }}>
                            <X size={20} color="grey" />
                        </button>
                    </div>
                ) : (
                    <>
                        <h1 style={{ margin: 0, textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)' }}>Вітаємо, {displayName}!</h1>
                        <button 
                            onClick={() => setIsEditing(true)} 
                            style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: '#a0a0a0' }}
                        >
                            <Edit2 size={20} />
                        </button>
                    </>
                )}
            </div>

            {/* КАРТКА СТАТИСТИКИ */}
            <div style={{ background: theme.colors.card, padding: '30px', borderRadius: '20px', boxShadow: '0 10px 25px rgba(0,0,0,0.3)' }}>
                <div style={{ marginBottom: '25px', display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <div style={{ background: theme.colors.background, padding: '10px', borderRadius: '10px' }}>
                        <Mail size={20} color="#60a5fa" />
                    </div>
                    <div>
                        <small style={{ color: '#94a3b8', display: 'block' }}>Прив'язана пошта</small>
                        <b style={{ color: theme.colors.textDark }}>{user.email}</b>
                    </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                    <div style={{ background: theme.colors.background, padding: '10px', borderRadius: '10px' }}>
                        <BarChart size={20} color="#4ade80" />
                    </div>
                    <div>
                        <small style={{ color: theme.colors.textMuted, display: 'block' }}>Всього записів</small>
                        <b style={{ color: theme.colors.textDark }}>{count} емоційних карток</b>
                    </div>
                </div>
            </div>
            
            <p style={{ textAlign: 'center', color: theme.colors.textMuted, fontSize: '12px', marginTop: '20px' }}>
                Дані автоматично синхронізуються з вашим акаунтом
            </p>
        </div>
    );
};

export default Profile;