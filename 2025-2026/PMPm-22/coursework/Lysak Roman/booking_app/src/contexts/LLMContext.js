import { createContext, useState, useContext } from 'react';

const LLMContext = createContext();

export const useLLM = () => {
  const context = useContext(LLMContext);
  if (!context) {
    throw new Error('useLLM must be used within LLMProvider');
  }
  return context;
};

export const LLMProvider = ({ children }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Привіт, я твій помічник з резервування. Чим я можу тобі допомогти?',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);

  const addMessage = (role, content) => {
    setMessages((prev) => [
      ...prev,
      {
        role,
        content,
        timestamp: new Date(),
      },
    ]);
  };

  const clearMessages = () => {
    setMessages([
      {
        role: 'assistant',
        content: 'Привіт, я твій помічник з резервування. Чим я можу тобі допомогти?',
        timestamp: new Date(),
      },
    ]);
  };

  const toggleChat = () => {
    setIsChatOpen((prev) => !prev);
  };

  const value = {
    messages,
    isLoading,
    isChatOpen,
    setIsLoading,
    addMessage,
    clearMessages,
    toggleChat,
  };

  return <LLMContext.Provider value={value}>{children}</LLMContext.Provider>;
};
