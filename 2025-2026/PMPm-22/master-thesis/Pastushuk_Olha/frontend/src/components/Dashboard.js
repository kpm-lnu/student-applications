import React, { useEffect, useState } from 'react';
import { supabase } from '../supabaseClient';
import { Trash2, Calendar, ArrowLeft, Check, X, Edit2, AlignLeft, PlayCircle } from 'lucide-react';
import { theme } from '../theme';

const Dashboard = ({ user }) => {
  const [entries, setEntries] = useState([]);
  const [selectedEntry, setSelectedEntry] = useState(null);
  
  // Стани для редагування назви
  const [isEditingTitle, setIsEditingTitle] = useState(false);
  const [tempTitle, setTempTitle] = useState('');

  // Стани для редагування опису
  const [isEditingDesc, setIsEditingDesc] = useState(false);
  const [tempDesc, setTempDesc] = useState('');

  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    fetchEntries();
  }, []);

  const fetchEntries = async () => {
    const { data, error } = await supabase
      .from('entries')
      .select('*')
      .order('created_at', { ascending: false });
    if (!error) setEntries(data);
  };

  const handleOpenDetails = (entry) => {
    setSelectedEntry(entry);
    setTempTitle(entry.title);
    setTempDesc(entry.description || '');
    setIsEditingTitle(false);
    setIsEditingDesc(false);
  };

  // Універсальна функція оновлення полів у базі
  const updateEntryField = async (field, value) => {
    setIsSaving(true);
    const { error } = await supabase
      .from('entries')
      .update({ [field]: value })
      .eq('id', selectedEntry.id);
    
    if (!error) {
      setSelectedEntry({ ...selectedEntry, [field]: value });
      if (field === 'title') setIsEditingTitle(false);
      if (field === 'description') setIsEditingDesc(false);
      fetchEntries(); // оновити список для головного екрану
    } else {
      alert("Помилка при збереженні");
    }
    setIsSaving(false);
  };

  const deleteEntry = async (id) => {
    if (!window.confirm("Видалити цей запис?")) return;
    const { error } = await supabase.from('entries').delete().eq('id', id);
    if (!error) {
      setSelectedEntry(null);
      fetchEntries();
    }
  };

  if (selectedEntry) {
    return (
      <div style={{ padding: '20px', maxWidth: '700px', margin: '0 auto', color: theme.colors.textDark }}>
        <button onClick={() => setSelectedEntry(null)} style={{ background: 'none', border: 'none', color: theme.colors.textDark, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '20px' }}>
          <ArrowLeft size={20} /> Назад
        </button>

        <div style={{ background: theme.colors.card, padding: '30px', borderRadius: '24px', boxShadow: '0 10px 30px rgba(0,0,0,0.3)' }}>
          
          {/* РЕДАГУВАННЯ НАЗВИ */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '25px' }}>
            {isEditingTitle ? (
              <div style={{ display: 'flex', gap: '10px', flex: 1 }}>
                <input 
                  value={tempTitle}
                  onChange={(e) => setTempTitle(e.target.value)}
                  autoFocus
                  style={{ flex: 1, fontSize: '24px', fontWeight: 'bold', background: theme.colors.background, border: '1px solid ' + theme.colors.primary, color: 'white', padding: '5px 10px', borderRadius: '10px' }}
                />
                <button onClick={() => updateEntryField('title', tempTitle)} style={{ background: theme.colors.green, border: 'none', padding: '8px', borderRadius: '50%', cursor: 'pointer' }}><Check size={20} color='black'/></button>
                <button onClick={() => { setIsEditingTitle(false); setTempTitle(selectedEntry.title); }} style={{ background: theme.colors.card, border: 'none', padding: '8px', borderRadius: '50%', cursor: 'pointer' }}><X size={20} color="black"/></button>
              </div>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                <h2 style={{ margin: 0 }}>{selectedEntry.title}</h2>
                <button onClick={() => setIsEditingTitle(true)} style={{ background: 'none', border: 'none', color: theme.colors.textMuted, cursor: 'pointer' }}><Edit2 size={18}/></button>
              </div>
            )}
            {!isEditingTitle && (
              <button onClick={() => deleteEntry(selectedEntry.id)} style={{ background: theme.colors.danger, border: 'none', padding: '10px', borderRadius: '50%', cursor: 'pointer' }}><Trash2 size={20} color="white" /></button>
            )}
          </div>

          <div style={{ color: theme.colors.textMuted, fontSize: '14px', marginBottom: '30px' }}>
             <Calendar size={14} style={{ verticalAlign: 'middle', marginRight: '5px', marginBottom: '5px' }}/> {new Date(selectedEntry.created_at).toLocaleString()}
          </div>

          <div style={{ background: theme.colors.background, padding: '20px', borderRadius: '15px', marginBottom: '30px' }}>
            <audio controls src={selectedEntry.audio_url} style={{ width: '100%' }} />
          </div>

          {/* VAD КАРТКИ */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', marginBottom: '30px' }}>
            <div style={{ background: theme.colors.card, padding: '15px', borderRadius: '12px', textAlign: 'center', color: theme.colors.textDark }}>
                <small style={{ color: '#4ade80' }}>Valence</small><br/><b>{selectedEntry.valence.toFixed(2)}</b>
            </div>
            <div style={{ background: theme.colors.card, padding: '15px', borderRadius: '12px', textAlign: 'center', color: theme.colors.textDark }}>
                <small style={{ color: '#60a5fa' }}>Arousal</small><br/><b>{selectedEntry.arousal.toFixed(2)}</b>
            </div>
            <div style={{ background: theme.colors.card, padding: '15px', borderRadius: '12px', textAlign: 'center', color: theme.colors.textDark }}>
                <small style={{ color: '#f87171' }}>Dominance</small><br/><b>{selectedEntry.dominance.toFixed(2)}</b>
            </div>
          </div>

          {/* РЕДАГУВАННЯ ОПИСУ */}
          <div style={{ borderTop: '1px solid #334155', paddingTop: '20px' }}>
            <p style={{ fontSize: '12px', color: theme.colors.textMuted, marginBottom: '10px', display: 'flex', alignItems: 'center', gap: '5px' }}>
              <AlignLeft size={14} /> НОТАТКИ ТА ОПИС
            </p>
            {isEditingDesc ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <textarea 
                  value={tempDesc}
                  onChange={(e) => setTempDesc(e.target.value)}
                  autoFocus
                  style={{ width: '100%', height: '100px', padding: '0px', borderRadius: '12px', border: '1px solid ' + theme.colors.primary, background: theme.colors.background, color: 'white', resize: 'none' }}
                />
                <div style={{ display: 'flex', gap: '10px', justifyContent: 'flex-end' }}>
                  <button onClick={() => { setIsEditingDesc(false); setTempDesc(selectedEntry.description || ''); }} style={{ background: theme.colors.card, color: 'white', border: 'none', padding: '8px 20px', borderRadius: '8px', cursor: 'pointer' }}>Скасувати</button>
                  <button onClick={() => updateEntryField('description', tempDesc)} disabled={isSaving} style={{ background: theme.colors.primary, color: theme.colors.textDark, border: 'none', padding: '8px 20px', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer' }}>Зберегти</button>
                </div>
              </div>
            ) : (
              <div 
                onClick={() => setIsEditingDesc(true)}
                style={{ minHeight: '60px', padding: '15px', background: theme.colors.background, borderRadius: '12px', cursor: 'pointer', border: '1px dashed #334155', color: selectedEntry.description ? 'white' : theme.colors.textMuted }}
              >
                {selectedEntry.description || "Натисніть, щоб додати опис..."}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto'}}>
      <h1 style={{ marginBottom: '30px', textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)', color: 'white'}}>Мій Журнал Емоцій</h1>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '15px'}}>
        {entries.map(entry => (
          <div 
            key={entry.id} 
            onClick={() => handleOpenDetails(entry)}
            style={{ 
              background: theme.colors.card, padding: '20px', borderRadius: '15px', cursor: 'pointer',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center'
            }}
            onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.02)'}
            onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
          > 
            <div>
              <h4 style={{ margin: '0 0 5px 0', color: theme.colors.textDark}}>{entry.title}</h4>
              <div style={{ display: 'flex', gap: '10px', fontSize: '12px', color: '#94a3b8' }}>
                <span style={{ color: '#4ade80' }}>V: {entry.valence.toFixed(2)}</span>
                <span>|</span>
                <span style={{ color: '#60a5fa' }}>A: {entry.arousal.toFixed(2)}</span>
                <span>|</span>
                <span style={{ color: '#f87171' }}>D: {entry.dominance.toFixed(2)}</span>
              </div>
            </div>
            <div style={{ background: theme.colors.primary, padding: '10px', borderRadius: '50%' }}><PlayCircle size={22} color="white"/></div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;