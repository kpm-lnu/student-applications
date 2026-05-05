import { User } from '../types';
import api from './api';

export interface ChatResult {
  text: string;
  appointmentChanged: boolean;
}

export async function sendChatMessage(
  history: { role: 'user' | 'model'; text: string }[],
  userMessage: string,
  _context: { user: User },
): Promise<ChatResult> {
  const response = await api.post<ChatResult>('/api/chat', {
    history,
    message: userMessage,
  });
  return response.data;
}
