import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { FluentProvider, webLightTheme } from '@fluentui/react-components';
import { MsalProvider } from '@azure/msal-react';
import { msalInstance } from './main';

import { AuthProvider, useAuth } from './contexts/AuthContext';
import { AppointmentsProvider } from './contexts/AppointmentsContext';
import { ChatProvider } from './contexts/ChatContext';
import { NotificationProvider } from './contexts/NotificationContext';

import { AppLayout } from './components/Layout/AppLayout';
import { AdminLayout } from './components/Layout/AdminLayout';
import { ChatWidget } from './components/ChatWidget/ChatWidget';

import { Home } from './pages/Home';
import { RoomDetail } from './pages/RoomDetail';
import { MyAppointments } from './pages/MyAppointments';
import { LoginPage } from './pages/LoginPage';

import { Dashboard } from './pages/admin/Dashboard';
import { UsersPage } from './pages/admin/UsersPage';
import { RoomsPage } from './pages/admin/RoomsPage';
import { RoomTypesPage } from './pages/admin/RoomTypesPage';
import { AddressesPage } from './pages/admin/AddressesPage';
import { StaffPage } from './pages/admin/StaffPage';
import { AppointmentsPage } from './pages/admin/AppointmentsPage';
import { StaffAppointmentsPage } from './pages/staff/StaffAppointmentsPage';
import { UserRole } from './types';

function AdminGuard({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user || user.role !== UserRole.Admin) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function StaffGuard({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user || user.role !== UserRole.Staff) return <Navigate to="/" replace />;
  return <>{children}</>;
}

function AuthGuard({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useAuth();
  if (isLoading) return null;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  return (
    <>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route element={<AppLayout />}>
          <Route path="/" element={<Home />} />
          <Route path="/rooms/:id" element={<RoomDetail />} />
          <Route
            path="/my-appointments"
            element={
              <AuthGuard>
                <MyAppointments />
              </AuthGuard>
            }
          />
          <Route
            path="/staff/appointments"
            element={
              <StaffGuard>
                <StaffAppointmentsPage />
              </StaffGuard>
            }
          />
          <Route
            path="/admin"
            element={
              <AdminGuard>
                <AdminLayout />
              </AdminGuard>
            }
          >
            <Route index element={<Dashboard />} />
            <Route path="users" element={<UsersPage />} />
            <Route path="rooms" element={<RoomsPage />} />
            <Route path="room-types" element={<RoomTypesPage />} />
            <Route path="addresses" element={<AddressesPage />} />
            <Route path="staff" element={<StaffPage />} />
            <Route path="appointments" element={<AppointmentsPage />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      <ChatWidget />
    </>
  );
}

export default function App() {
  return (
    <MsalProvider instance={msalInstance}>
      <FluentProvider theme={webLightTheme}>
        <AuthProvider>
          <NotificationProvider>
            <AppointmentsProvider>
              <ChatProvider>
                <BrowserRouter>
                  <AppRoutes />
                </BrowserRouter>
              </ChatProvider>
            </AppointmentsProvider>
          </NotificationProvider>
        </AuthProvider>
      </FluentProvider>
    </MsalProvider>
  );
}
