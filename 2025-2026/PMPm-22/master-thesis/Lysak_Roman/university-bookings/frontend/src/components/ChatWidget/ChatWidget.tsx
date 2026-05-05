import React, { useEffect, useRef, useState } from 'react';
import { makeStyles, Spinner } from '@fluentui/react-components';
import {
  ChatRegular,
  DismissRegular,
  SendRegular,
  BotRegular,
} from '@fluentui/react-icons';
import { useChat } from '../../contexts/ChatContext';
import { useAuth } from '../../contexts/AuthContext';

const useStyles = makeStyles({
  fab: {
    position: 'fixed',
    bottom: '28px',
    right: '28px',
    zIndex: 1000,
    width: '54px',
    height: '54px',
    borderRadius: '50%',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 4px 20px rgba(20, 27, 77, 0.35)',
    transition: 'background 0.2s, transform 0.15s',
    ':hover': {
      backgroundColor: 'var(--color-secondary-2)',
      transform: 'scale(1.07)',
    },
  },
  panel: {
    position: 'fixed',
    bottom: '94px',
    right: '28px',
    width: '370px',
    maxHeight: 'calc(100vh - 120px)',
    height: '520px',
    backgroundColor: 'var(--color-white)',
    borderRadius: 'var(--radius-lg)',
    boxShadow: '0 8px 48px rgba(20, 27, 77, 0.22)',
    display: 'flex',
    flexDirection: 'column',
    zIndex: 1000,
    overflow: 'hidden',
    border: '1px solid rgba(20,27,77,0.10)',
  },
  header: {
    backgroundColor: 'var(--color-primary)',
    padding: '14px 18px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: '0',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  headerIcon: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    backgroundColor: 'rgba(255,255,255,0.15)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'var(--color-white)',
    flexShrink: '0',
  },
  headerTitle: {
    fontFamily: 'var(--font-brand)',
    fontWeight: '700',
    fontSize: '15px',
    color: 'var(--color-white)',
    display: 'block',
    lineHeight: '1.2',
  },
  headerSub: {
    fontFamily: 'var(--font-brand)',
    fontSize: '11px',
    color: 'rgba(255,255,255,0.60)',
    display: 'block',
    marginTop: '1px',
  },
  closeBtn: {
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: 'rgba(255,255,255,0.70)',
    width: '30px',
    height: '30px',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'background 0.15s',
    ':hover': {
      backgroundColor: 'rgba(255,255,255,0.15)',
      color: 'var(--color-white)',
    },
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '16px 14px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    backgroundColor: '#F7F8FC',
  },
  bubble: {
    maxWidth: '82%',
    padding: '10px 14px',
    borderRadius: 'var(--radius-md)',
    fontFamily: 'var(--font-brand)',
    fontSize: '13.5px',
    lineHeight: '1.55',
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
  },
  userBubble: {
    alignSelf: 'flex-end',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    borderBottomRightRadius: 'var(--radius-sm)',
  },
  assistantBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'var(--color-white)',
    color: 'var(--color-text)',
    borderBottomLeftRadius: 'var(--radius-sm)',
    boxShadow: '0 1px 4px rgba(20,27,77,0.08)',
    border: '1px solid rgba(20,27,77,0.06)',
  },
  typingBubble: {
    alignSelf: 'flex-start',
    backgroundColor: 'var(--color-white)',
    borderBottomLeftRadius: 'var(--radius-sm)',
    boxShadow: '0 1px 4px rgba(20,27,77,0.08)',
    border: '1px solid rgba(20,27,77,0.06)',
    padding: '10px 14px',
    borderRadius: 'var(--radius-md)',
  },
  inputArea: {
    padding: '10px 12px 12px',
    borderTop: '1px solid rgba(20,27,77,0.08)',
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-end',
    backgroundColor: 'var(--color-white)',
    flexShrink: '0',
  },
  textarea: {
    flex: 1,
    fontFamily: 'var(--font-brand)',
    fontSize: '13.5px',
    border: '1.5px solid rgba(20,27,77,0.18)',
    borderRadius: 'var(--radius-md)',
    padding: '9px 12px',
    resize: 'none',
    outline: 'none',
    color: 'var(--color-text)',
    backgroundColor: '#F7F8FC',
    lineHeight: '1.5',
    transition: 'border-color 0.15s',
  },
  sendBtn: {
    width: '38px',
    height: '38px',
    borderRadius: 'var(--radius-md)',
    backgroundColor: 'var(--color-primary)',
    color: 'var(--color-white)',
    border: 'none',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: '0',
    transition: 'background 0.15s',
    ':disabled': {
      opacity: '0.4',
      cursor: 'not-allowed',
    },
  },
});

export function ChatWidget() {
  const styles = useStyles();
  const { user } = useAuth();
  const { messages, isOpen, isTyping, openChat, closeChat, sendMessage } = useChat();
  const [input, setInput] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const fabRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    if (!isOpen) return;
    const handleMouseDown = (e: MouseEvent) => {
      if (
        panelRef.current?.contains(e.target as Node) ||
        fabRef.current?.contains(e.target as Node)
      ) return;
      closeChat();
    };
    document.addEventListener('mousedown', handleMouseDown);
    return () => document.removeEventListener('mousedown', handleMouseDown);
  }, [isOpen, closeChat]);

  if (!user) return null;

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isTyping) return;
    setInput('');
    await sendMessage(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const canSend = !!input.trim() && !isTyping;

  return (
    <>
      <button
        ref={fabRef}
        className={styles.fab}
        onClick={isOpen ? closeChat : openChat}
        title="AI Асистент"
      >
        {isOpen
          ? <DismissRegular fontSize={22} />
          : <ChatRegular fontSize={22} />
        }
      </button>

      {isOpen && (
        <div ref={panelRef} className={styles.panel}>
          <div className={styles.header}>
            <div className={styles.headerLeft}>
              <div className={styles.headerIcon}>
                <BotRegular fontSize={20} />
              </div>
              <div>
                <span className={styles.headerTitle}>AI Асистент</span>
                <span className={styles.headerSub}>Gemini · Завжди онлайн</span>
              </div>
            </div>
            <button className={styles.closeBtn} onClick={closeChat}>
              <DismissRegular fontSize={16} />
            </button>
          </div>

          <div className={styles.messages}>
            {messages.length === 0 && (
              <div className={`${styles.bubble} ${styles.assistantBubble}`}>
                {`Привіт, ${user.displayName.split(' ')[1] ?? user.displayName.split(' ')[0]}! 👋\nЯ можу допомогти вам знайти потрібну аудиторію, переглянути записи або відповісти на питання про систему.`}
              </div>
            )}

            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`${styles.bubble} ${msg.role === 'user' ? styles.userBubble : styles.assistantBubble}`}
              >
                {msg.content}
              </div>
            ))}

            {isTyping && (
              <div className={styles.typingBubble}>
                <Spinner size="tiny" label="Пишу відповідь..." labelPosition="after" />
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <div className={styles.inputArea}>
            <textarea
              className={styles.textarea}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Напишіть повідомлення..."
              rows={2}
              onFocus={(e) => (e.currentTarget.style.borderColor = 'var(--color-secondary-3)')}
              onBlur={(e) => (e.currentTarget.style.borderColor = 'rgba(20,27,77,0.18)')}
            />
            <button
              className={styles.sendBtn}
              onClick={handleSend}
              disabled={!canSend}
              title="Надіслати"
              style={{ opacity: canSend ? 1 : 0.38 }}
              onMouseOver={(e) => { if (canSend) e.currentTarget.style.backgroundColor = 'var(--color-secondary-2)'; }}
              onMouseOut={(e) => { e.currentTarget.style.backgroundColor = 'var(--color-primary)'; }}
            >
              <SendRegular fontSize={17} />
            </button>
          </div>
        </div>
      )}
    </>
  );
}
