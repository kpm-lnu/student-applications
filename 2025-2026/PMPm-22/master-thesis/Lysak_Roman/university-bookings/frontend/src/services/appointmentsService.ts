import api from './api';
import {
  Appointment,
  CreateAppointmentRequest,
  UpdateAppointmentStatusRequest,
} from '../types';

export const appointmentsService = {
  getMy: () =>
    api.get<Appointment[]>('/api/appointments/my').then((r) => r.data),

  create: (body: CreateAppointmentRequest) =>
    api.post<Appointment>('/api/appointments', body).then((r) => r.data),

  cancel: (id: string, reason?: string) =>
    api
      .delete(`/api/appointments/${id}`, { data: { reason } })
      .then((r) => r.data),

  // Staff only
  getStaff: (params?: { status?: string }) =>
    api.get<Appointment[]>('/api/staff/appointments', { params }).then((r) => r.data),

  updateStaffStatus: (id: string, body: UpdateAppointmentStatusRequest) =>
    api.patch<Appointment>(`/api/staff/appointments/${id}/status`, body).then((r) => r.data),

  // Admin only
  getAll: (params?: {
    status?: string;
    from?: string;
    to?: string;
    roomId?: string;
  }) =>
    api
      .get<Appointment[]>('/api/admin/appointments', { params })
      .then((r) => r.data),

  updateStatus: (id: string, body: UpdateAppointmentStatusRequest) =>
    api
      .patch<Appointment>(`/api/appointments/${id}/status`, body)
      .then((r) => r.data),
};
