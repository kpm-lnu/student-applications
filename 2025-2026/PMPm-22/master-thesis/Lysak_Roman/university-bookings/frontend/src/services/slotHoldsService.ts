import api, { cachedAccessToken } from './api';
import type { SlotHoldResponse } from '../types';

export const slotHoldsService = {
  async create(roomId: string, startDateTime: string, endDateTime: string): Promise<SlotHoldResponse> {
    const { data } = await api.post('/api/slot-holds', { roomId, startDateTime, endDateTime });
    return data;
  },

  async release(holdId: string): Promise<void> {
    await api.delete(`/api/slot-holds/${holdId}`);
  },

  releaseSync(holdId: string): void {
    const baseUrl = (import.meta.env.VITE_API_URL as string | undefined) ?? '';
    const headers: HeadersInit = { 'Content-Type': 'application/json' };
    if (cachedAccessToken) headers['Authorization'] = `Bearer ${cachedAccessToken}`;
    fetch(`${baseUrl}/api/slot-holds/${holdId}`, { method: 'DELETE', keepalive: true, headers });
  },
};
