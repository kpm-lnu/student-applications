import api from './api';
import { Address } from '../types';

export const addressesService = {
  getAll: () =>
    api.get<Address[]>('/api/admin/addresses').then((r) => r.data),

  create: (body: { street: string }) =>
    api.post<Address>('/api/admin/addresses', body).then((r) => r.data),

  update: (id: string, body: { street: string }) =>
    api.put<Address>(`/api/admin/addresses/${id}`, body).then((r) => r.data),

  delete: (id: string) =>
    api.delete(`/api/admin/addresses/${id}`).then((r) => r.data),
};
