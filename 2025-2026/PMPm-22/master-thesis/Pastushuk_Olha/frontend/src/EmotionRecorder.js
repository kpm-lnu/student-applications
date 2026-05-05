import React, { useState } from 'react';
import { useReactMediaRecorder } from "react-media-recorder";
import axios from 'axios';
import { supabase } from './supabaseClient';
import { Mic, Square, Save } from 'lucide-react';
import { theme } from './theme';

const EmotionRecorder = ({ user }) => {
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [lastResult, setLastResult] = useState(null);
    const [currentEntryId, setCurrentEntryId] = useState(null);
    const [title, setTitle] = useState('');
    const [isUpdating, setIsUpdating] = useState(false);

    const { status, startRecording, stopRecording } = useReactMediaRecorder({ 
        audio: true, 
        onStop: (blobUrl, blob) => handleFullProcess(blob) 
    });

    const handleFullProcess = async (blob) => {
        setIsAnalyzing(true);
        setLastResult(null);
        
        try {
            const formData = new FormData();
            formData.append('file', blob, 'record.wav');
            const response = await axios.post(`http://localhost:8000/analyze`, formData);
            const vad = response.data;

            const filePath = `entries/${user.id}/${Date.now()}.wav`;
            await supabase.storage.from('recordings').upload(filePath, blob);
            const { data: { publicUrl } } = supabase.storage.from('recordings').getPublicUrl(filePath);

            const tempTitle = `Запис від ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
            const { data: newEntry, error } = await supabase.from('entries').insert([{
                title: tempTitle,
                audio_url: publicUrl,
                valence: vad.valence,
                arousal: vad.arousal,
                dominance: vad.dominance,
                interpretation: vad.interpretation,
                user_id: user.id
            }]).select().single();

            if (error) throw error;

            setLastResult(vad);
            setCurrentEntryId(newEntry.id);
            setTitle(tempTitle);

        } catch (err) {
            console.error("Помилка:", err);
            alert("Помилка при обробці. Перевір консоль бекенду.");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const finalizeRename = async () => {
        setIsUpdating(true);
        try {
            const { error } = await supabase
                .from('entries')
                .update({ title: title })
                .eq('id', currentEntryId);

            if (error) throw error;

            setLastResult(null);
            setCurrentEntryId(null);
            setTitle('');
            alert("Запис успішно додано у щоденник!");
        } catch (err) {
            console.error("Помилка оновлення назви:", err);
        } finally {
            setIsUpdating(false);
        }
    };

    return (
        <div style={{ padding: '20px', textAlign: 'center', color: 'white' }}>
            {/* ЕКРАН 1: МІКРОФОН ТА ПЛАШКА */}
            {!lastResult && !isAnalyzing && (
                <div style={{ marginTop: '50px' }}>
                    <h1 style={{ textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)' }}>Запишіть свої емоції</h1>
                    
                    {/* БІЛИЙ ПРЯМОКУТНИК (Плашка) */}
                    <div style={{ 
                        background: 'rgba(255, 255, 255, 0.85)', 
                        padding: '40px',
                        borderRadius: theme.borderRadius,
                        maxWidth: '400px',
                        margin: '40px auto',
                        boxShadow: '0 10px 30px rgba(0,0,0,0.15)',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        gap: '20px',
                        backdropFilter: 'blur(5px)' // Ефект матового скла
                    }}>
                        {status !== "recording" ? (
                            <button 
                                onClick={startRecording} 
                                style={{ 
                                    background: theme.colors.danger, 
                                    borderRadius: '50%', 
                                    padding: '25px', 
                                    cursor: 'pointer', 
                                    border: 'none',
                                    boxShadow: '0 4px 15px rgba(239, 68, 68, 0.3)',
                                    transition: theme.transition
                                }}
                            >
                                <Mic size={48} color="white" />
                            </button>
                        ) : (
                            <button 
                                onClick={stopRecording} 
                                style={{ 
                                    background: theme.colors.card, 
                                    borderRadius: '50%', 
                                    padding: '25px', 
                                    cursor: 'pointer', 
                                    border: 'none',
                                    transition: theme.transition
                                }}
                            >
                                <Square size={48} color="white" />
                            </button>
                        )}

                        <p style={{ 
                            color: '#1e293b', 
                            margin: 0,
                            fontWeight: '600',
                            fontSize: '18px'
                        }}>
                            {status === "recording" ? "Слухаю вас..." : "Натисніть для запису"}
                        </p>
                    </div>
                </div>
            )}

            {/* ЕКРАН 2: АНАЛІЗ */}
            {isAnalyzing && (
                <div style={{ marginTop: '100px' }}>
                    <div style={{ fontSize: '40px', marginBottom: '20px' }}>⚡</div>
                    <p style={{ fontSize: '18px', fontWeight: '500', color: theme.colors.textDark }}>Нейронка працює, зачекайте...</p>
                </div>
            )}

            {/* ЕКРАН 3: РЕЗУЛЬТАТИ ТА ПЕРЕЙМЕНУВАННЯ */}
            {lastResult && (
                <div style={{ 
                    marginTop: '20px', 
                    background: theme.colors.card, 
                    padding: '30px', 
                    borderRadius: theme.borderRadius, 
                    maxWidth: '450px', 
                    margin: '20px auto',
                    boxShadow: '0 10px 30px rgba(0,0,0,0.3)'
                }}>
                    <h3 style={{ color: theme.colors.textDark, marginBottom: '20px' }}>
                        Результат: {lastResult.interpretation}
                    </h3>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', margin: '20px 0', fontSize: '14px' }}>
                        <div style={{ background: theme.colors.background, padding: '15px', borderRadius: '12px' }}>
                            <small style={{ color: theme.colors.textMuted }}>Valence</small><br/>
                            <b style={{ fontSize: '16px', color: '#4ade80' }}>{lastResult.valence.toFixed(2)}</b>
                        </div>
                        <div style={{ background: theme.colors.background, padding: '15px', borderRadius: '12px' }}>
                            <small style={{ color: theme.colors.textMuted }}>Arousal</small><br/>
                            <b style={{ fontSize: '16px', color: '#60a5fa' }}>{lastResult.arousal.toFixed(2)}</b>
                        </div>
                        <div style={{ background: theme.colors.background, padding: '15px', borderRadius: '12px' }}>
                            <small style={{ color: theme.colors.textMuted }}>Dominance</small><br/>
                            <b style={{ fontSize: '16px', color: '#f87171' }}>{lastResult.dominance.toFixed(2)}</b>
                        </div>
                    </div>

                    <div style={{ marginTop: '30px', textAlign: 'left', color: theme.colors.textDark }}>
                        <label style={{ display: 'block', marginBottom: '8px', fontSize: '11px', fontWeight: 'bold', color: theme.colors.textMuted, letterSpacing: '1px' }}>
                            ЯК НАЗВАТИ ЦЕЙ ЗАПИС?
                        </label>
                        <input 
                            type="text" 
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder="Наприклад: Ранкова кава..."
                            style={{ 
                                width: '100%', 
                                padding: '14px', 
                                borderRadius: '10px', 
                                border: `1px solid ${theme.colors.textMuted}44`, 
                                background: theme.colors.background, 
                                color: theme.colors.textDark, 
                                marginBottom: '20px', 
                                boxSizing: 'border-box' 
                            }}
                        />
                        <button 
                            onClick={finalizeRename} 
                            disabled={isUpdating}
                            style={{ 
                                width: '100%', 
                                background: theme.colors.primary, 
                                color: theme.colors.textDark, 
                                padding: '16px', 
                                borderRadius: '12px', 
                                fontWeight: 'bold', 
                                border: 'none', 
                                cursor: 'pointer', 
                                display: 'flex', 
                                alignItems: 'center', 
                                justifyContent: 'center', 
                                gap: '10px',
                                boxShadow: `0 4px 14px ${theme.colors.primary}44`
                            }}
                        >
                            <Save size={20} /> {isUpdating ? "Зберігаємо..." : "Готово, до нового запису"}
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default EmotionRecorder;