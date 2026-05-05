// ─── Enums ───────────────────────────────────────────────────────────────────

export enum SlotMode {
  Interval = 'Interval',
  Para = 'Para',
}

export const UNIVERSITY_PARA = [
  { index: 1, label: '1 пара', start: '08:30', end: '09:50' },
  { index: 2, label: '2 пара', start: '10:10', end: '11:30' },
  { index: 3, label: '3 пара', start: '11:50', end: '13:10' },
  { index: 4, label: '4 пара', start: '13:30', end: '14:50' },
  { index: 5, label: '5 пара', start: '15:05', end: '16:25' },
  { index: 6, label: '6 пара', start: '16:40', end: '18:00' },
  { index: 7, label: '7 пара', start: '18:10', end: '19:30' },
  { index: 8, label: '8 пара', start: '19:40', end: '21:00' },
] as const;

export enum UserRole {
  Student = 'Student',
  Staff = 'Staff',
  Admin = 'Admin',
}

export enum AppointmentStatus {
  Pending = 'Pending',
  Confirmed = 'Confirmed',
  Cancelled = 'Cancelled',
  Completed = 'Completed',
}

export enum NotificationType {
  Email = 'Email',
  Teams = 'Teams',
  Calendar = 'Calendar',
}

export enum NotificationStatus {
  Sent = 'Sent',
  Failed = 'Failed',
}

// ─── Room ─────────────────────────────────────────────────────────────────────

export interface RoomType {
  id: string;
  name: string;   // slug(Ярлик): "classroom", "sport", "conference"
  label: string;  // display: "Аудиторія", "Спортивний зал", etc.
}

export interface Address {
  id: string;
  street: string;
}

export interface Room {
  id: string;
  name: string;
  roomNumber?: string;
  roomType: RoomType | null;
  address: Address | null;
  description?: string;
  capacity?: number;
  isActive: boolean;
  responsiblePerson?: {
    id: string;
    displayName: string;
  };
  availability: Availability[];
  slotMode: SlotMode;
}

// ─── Duration ─────────────────────────────────────────────────────────────────

export const DURATION_OPTIONS = [
  { value: 40,  label: 'Академічна година (40 хв)' },
  { value: 60,  label: '1 година' },
  { value: 80,  label: '2 академічні години (1 год 20 хв)' },
  { value: 120, label: '2 години' },
] as const;

export type DurationMinutes = 40 | 60 | 80 | 120;

// ─── Core Entities ────────────────────────────────────────────────────────────

export interface User {
  id: string;
  azureObjectId: string;
  email: string;
  displayName: string;
  role: UserRole;
  createdAt: string;
  lastLoginAt: string;
}

export interface StaffMember {
  id: string;
  userId: string;
  displayName: string;
  email: string;
}

export interface Availability {
  id: string;
  roomId: string;
  dayOfWeek: number; // 0 = Sunday … 6 = Saturday
  startTime: string; // "HH:mm"
  endTime: string;   // "HH:mm"
  availableParaIndices: number[];
}

export interface Appointment {
  id: string;
  roomId: string;
  roomName: string;
  roomType: string;  // label string from backend
  clientUserId: string;
  clientUser?: User;
  startDateTime: string;     // ISO 8601 UTC
  endDateTime: string;       // ISO 8601 UTC
  durationMinutes: DurationMinutes;
  status: AppointmentStatus;
  notes?: string | null;
  createdAt: string;
  cancelledAt?: string | null;
  cancellationReason?: string | null;
}

export interface TimeSlot {
  startTime: string; // ISO 8601 UTC
  endTime: string;
  available: boolean;
  isHeld: boolean;
}

export interface SlotHoldResponse {
  id: string;
  expiresAt: string; // ISO 8601
}

// ─── DTOs / Request Bodies ────────────────────────────────────────────────────

export interface CreateAppointmentRequest {
  roomId: string;
  durationMinutes: DurationMinutes;
  startDateTime: string; // ISO 8601 UTC
  notes?: string;
}

export interface CancelAppointmentRequest {
  reason?: string;
}

export interface UpdateRoleRequest {
  role: UserRole;
}

export interface UpdateAppointmentStatusRequest {
  status: AppointmentStatus;
}

// ─── Admin Stats ──────────────────────────────────────────────────────────────

export interface DashboardStats {
  totalBookingsToday: number;
  totalBookingsThisMonth: number;
  pendingAppointments: number;
  cancellationRate: number; // 0–100 percentage
  popularRooms: PopularRoom[];
  bookingsPerDay: BookingsPerDay[];
}

export interface PopularRoom {
  roomId: string;
  roomName: string;
  count: number;
}

export interface BookingsPerDay {
  date: string; // "YYYY-MM-DD"
  count: number;
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

export interface AuthProfile {
  user: User;
  token: string;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}
