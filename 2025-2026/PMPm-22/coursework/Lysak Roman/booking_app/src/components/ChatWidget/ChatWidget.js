import { useState, useEffect, useRef } from 'react';
import { useLLM } from '../../contexts/LLMContext';
import { useBookings } from '../../contexts/BookingsContext';
import { sendMessageToGemini, isApiKeyConfigured } from '../../services/geminiService';
import ReactMarkdown from 'react-markdown';
import {
  ChatContainer,
  ChatButton,
  ChatWindow,
  ChatHeader,
  ChatTitle,
  CloseButton,
  MessagesContainer,
  Message,
  MessageBubble,
  MessageTime,
  InputContainer,
  Input,
  SendButton,
  LoadingDots,
  ApiKeyWarning,
} from './ChatWidget_Styled';

const ChatWidget = () => {
  const { messages, isLoading, isChatOpen, setIsLoading, addMessage, toggleChat } = useLLM();
  const { bookings, appointments, loading: bookingsLoading, deleteAppointment } = useBookings();
  const [inputValue, setInputValue] = useState('');
  const messagesEndRef = useRef(null);
  const apiKeyConfigured = isApiKeyConfigured();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading || !apiKeyConfigured) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    addMessage('user', userMessage);
    setIsLoading(true);

    try {
      const response = await sendMessageToGemini(userMessage, bookings, appointments, deleteAppointment);
      addMessage('assistant', response);
    } catch (error) {
      addMessage('assistant', error.message || 'Вибачте, виникла помилка. Спробуйте ще раз.');
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('uk-UA', {
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <ChatContainer>
      {isChatOpen && (
        <ChatWindow>
          <ChatHeader>
            <ChatTitle>Помічник з резервування</ChatTitle>
            <CloseButton onClick={toggleChat}>×</CloseButton>
          </ChatHeader>

          {!apiKeyConfigured && (
            <ApiKeyWarning>
              <strong>⚠️ API ключ не налаштований</strong>
              Будь ласка, додайте ваш Gemini API ключ до файлу <code>.env.local</code>:
              <br />
              <code>REACT_APP_GEMINI_API_KEY=ваш_ключ</code>
              <br />
              Отримайте ключ: <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer">Google AI Studio</a>
            </ApiKeyWarning>
          )}

          <MessagesContainer>
            {messages.map((message, index) => (
              <Message key={index} $isUser={message.role === 'user'}>
                <MessageBubble $isUser={message.role === 'user'}>
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </MessageBubble>
                <MessageTime>{formatTime(message.timestamp)}</MessageTime>
              </Message>
            ))}
            {isLoading && (
              <Message $isUser={false}>
                <MessageBubble $isUser={false}>
                  <LoadingDots>
                    <span />
                    <span />
                    <span />
                  </LoadingDots>
                </MessageBubble>
              </Message>
            )}
            <div ref={messagesEndRef} />
          </MessagesContainer>

          <InputContainer onSubmit={handleSubmit}>
            <Input
              type="text"
              placeholder={apiKeyConfigured ? "Напишіть ваше запитання..." : "API ключ не налаштований"}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={isLoading || !apiKeyConfigured || bookingsLoading}
            />
            <SendButton type="submit" disabled={isLoading || !inputValue.trim() || !apiKeyConfigured}>
              ➤
            </SendButton>
          </InputContainer>
        </ChatWindow>
      )}

      <ChatButton onClick={toggleChat} title="Відкрити чат помічника">
        {isChatOpen ? '✕' : '💬'}
      </ChatButton>
    </ChatContainer>
  );
};

export default ChatWidget;
