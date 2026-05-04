import api from './api';
import { Room, TimeSlot, DurationMinutes, SlotMode } from '../types';

export const roomsService = {
  getAll: (type?: string) =>
    api
      .get<Room[]>('/api/rooms', { params: type ? { type } : undefined })
      .then((r) => r.data),

  getById: (id: string) =>
    api.get<Room>(`/api/rooms/${id}`).then((r) => r.data),

  getAvailableSlots: (roomId: string, date: string, duration: DurationMinutes) =>
    api
      .get<TimeSlot[]>(`/api/rooms/${roomId}/available-slots`, {
        params: { date, duration },
      })
      .then((r) => r.data),

  // Admin only
  adminGetAll: () =>
    api.get<Room[]>('/api/admin/rooms').then((r) => r.data),

  adminCreate: (body: {
    name: string;
    roomNumber?: string;
    roomTypeId?: string;
    addressId?: string;
    description?: string;
    capacity?: number;
    isActive: boolean;
    responsiblePersonId?: string;
    slotMode: SlotMode;
  }) => api.post<Room>('/api/admin/rooms', body).then((r) => r.data),

  adminUpdate: (
    id: string,
    body: {
      name: string;
      roomNumber?: string;
      roomTypeId?: string;
      addressId?: string;
      description?: string;
      capacity?: number;
      isActive: boolean;
      responsiblePersonId?: string;
      slotMode: SlotMode;
    },
  ) => api.put<Room>(`/api/admin/rooms/${id}`, body).then((r) => r.data),

  adminDelete: (id: string) =>
    api.delete(`/api/admin/rooms/${id}`).then((r) => r.data),

  adminAddAvailability: (
    roomId: string,
    body: { dayOfWeek: number; startTime: string; endTime: string; availableParaIndices?: number[] },
  ) =>
    api
      .post(`/api/admin/rooms/${roomId}/availability`, body)
      .then((r) => r.data),

  adminDeleteAvailability: (roomId: string, availId: string) =>
    api
      .delete(`/api/admin/rooms/${roomId}/availability/${availId}`)
      .then((r) => r.data),
};
