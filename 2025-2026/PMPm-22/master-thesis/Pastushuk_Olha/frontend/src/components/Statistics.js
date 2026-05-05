import React, { useEffect, useState } from 'react';
import { supabase } from '../supabaseClient';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { TrendingUp, Activity, PieChart as PieIcon } from 'lucide-react';
import { theme } from '../theme';

const Statistics = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    const { data: entries, error } = await supabase
      .from('entries')
      .select('created_at, valence, arousal, dominance')
      .order('created_at', { ascending: true });

    if (!error) {
      // Форматуємо дату для графіка
      const formattedData = entries.map(e => ({
        date: new Date(e.created_at).toLocaleDateString('uk-UA', { day: '2-digit', month: '2-digit' }),
        Valence: parseFloat(e.valence.toFixed(2)),
        Arousal: parseFloat(e.arousal.toFixed(2)),
        Dominance: parseFloat(e.dominance.toFixed(2)),
      }));
      setData(formattedData);
    }
    setLoading(false);
  };

  const getAverage = (key) => {
    if (data.length === 0) return 0;
    return (data.reduce((acc, curr) => acc + curr[key], 0) / data.length).toFixed(2);
  };

  if (loading) return <div style={{ textAlign: 'center', padding: '50px' }}>Завантаження статистики...</div>;

  return (
    <div style={{ padding: '20px', maxWidth: '900px', margin: '0 auto', color: 'white' }}>
      <h1 style={{ marginBottom: '30px', textAlign: 'center', textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)' }}>Аналітика емоцій</h1>

      {/* КАРТКИ ШВИДКОЇ СТАТИСТИКИ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
        <div style={{ background: theme.colors.card, padding: '20px', borderRadius: '16px', textAlign: 'center', borderBottom: '4px solid #4ade80' }}>
          <small style={{ color: theme.colors.textMuted }}>Середня Приємність</small>
          <h3 style={{ margin: '10px 0 0 0', color: '#4ade80' }}>{getAverage('Valence')}</h3>
        </div>
        <div style={{ background: theme.colors.card, padding: '20px', borderRadius: '16px', textAlign: 'center', borderBottom: '4px solid #60a5fa' }}>
          <small style={{ color: theme.colors.textMuted }}>Середнє Збудження</small>
          <h3 style={{ margin: '10px 0 0 0', color: '#60a5fa' }}>{getAverage('Arousal')}</h3>
        </div>
        <div style={{ background: theme.colors.card, padding: '20px', borderRadius: '16px', textAlign: 'center', borderBottom: '4px solid #f87171' }}>
          <small style={{ color: theme.colors.textMuted }}>Середній Контроль</small>
          <h3 style={{ margin: '10px 0 0 0', color: '#f87171' }}>{getAverage('Dominance')}</h3>
        </div>
      </div>

      {/* ГРАФІК */}
      <div style={{ background: theme.colors.card, padding: '30px 20px 20px 10px', borderRadius: '24px', height: '400px', boxShadow: '0 10px 30px rgba(0,0,0,0.3)' }}>
        <h4 style={{ color: theme.colors.textDark, marginLeft: '40px', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Activity size={18} color="#4ade80" /> Динаміка станів по датах
        </h4>
        <ResponsiveContainer width="95%" height="80%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
            <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} tickMargin={10} />
            <YAxis stroke="#94a3b8" fontSize={12} domain={[-1, 1]} />
            <Tooltip 
              contentStyle={{ background: '#0f172a', border: '1px solid #334155', borderRadius: '10px' }}
              itemStyle={{ fontSize: '12px' }}
            />
            <Legend verticalAlign="top" height={36}/>
            <Line type="monotone" dataKey="Valence" stroke="#4ade80" strokeWidth={3} dot={{ r: 4 }} activeDot={{ r: 8 }} />
            <Line type="monotone" dataKey="Arousal" stroke="#60a5fa" strokeWidth={3} dot={{ r: 4 }} />
            <Line type="monotone" dataKey="Dominance" stroke="#f87171" strokeWidth={3} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Statistics;