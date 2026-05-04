import React, {
  createContext,
  useContext,
  useState,
  useCallback,
} from 'react';
import { ChatMessage } from '../types';
import { sendChatMessage } from '../services/geminiService';
import { useAuth } from './AuthContext';
import { useAppointments } from './AppointmentsContext';

interface ChatContextValue {
  messages: ChatMessage[];
  isOpen: boolean;
  isTyping: boolean;
  openChat: () => void;
  closeChat: () => void;
  sendMessage: (text: string) => Promise<void>;
  clearMessages: () => void;
}

const ChatContext = createContext<ChatContextValue>({
  messages: [],
  isOpen: false,
  isTyping: false,
  openChat: () => {},
  closeChat: () => {},
  sendMessage: async () => {},
  clearMessages: () => {},
});

export function ChatProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const { refresh } = useAppointments();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const openChat = useCallback(() => setIsOpen(true), []);

  const closeChat = () => setIsOpen(false);

  const clearMessages = () => setMessages([]);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!user) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'user',
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsTyping(true);

      // Build history for Gemini (exclude the just-added user message)
      const history = messages.map((m) => ({
        role: m.role === 'user' ? ('user' as const) : ('model' as const),
        text: m.content,
      }));

      try {
        const result = await sendChatMessage(history, text, { user });
        const assistantMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.text,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, assistantMsg]);
        if (result.appointmentChanged) refresh();
      } catch (err) {
        console.error('Chat error', err);
        const errMsg: ChatMessage = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content:
            'Вибачте, сталася помилка. Будь ласка, спробуйте ще раз.',
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errMsg]);
      } finally {
        setIsTyping(false);
      }
    },
    [user, messages],
  );

  return (
    <ChatContext.Provider
      value={{
        messages,
        isOpen,
        isTyping,
        openChat,
        closeChat,
        sendMessage,
        clearMessages,
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export const useChat = () => useContext(ChatContext);
