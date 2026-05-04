import api from './api';
import { RoomType } from '../types';

export const roomTypesService = {
  getAll: () =>
    api.get<RoomType[]>('/api/room-types').then((r) => r.data),

  adminGetAll: () =>
    api.get<RoomType[]>('/api/admin/room-types').then((r) => r.data),

  create: (body: { name: string; label: string }) =>
    api.post<RoomType>('/api/admin/room-types', body).then((r) => r.data),

  update: (id: string, body: { name: string; label: string }) =>
    api.put<RoomType>(`/api/admin/room-types/${id}`, body).then((r) => r.data),

  delete: (id: string) =>
    api.delete(`/api/admin/room-types/${id}`).then((r) => r.data),
};
