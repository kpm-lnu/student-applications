import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from 'react';
import { Appointment } from '../types';
import { appointmentsService } from '../services/appointmentsService';
import { useAuth } from './AuthContext';

interface AppointmentsContextValue {
  appointments: Appointment[];
  isLoading: boolean;
  refresh: () => Promise<void>;
}

const AppointmentsContext = createContext<AppointmentsContextValue>({
  appointments: [],
  isLoading: false,
  refresh: async () => {},
});

export function AppointmentsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { isAuthenticated } = useAuth();
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!isAuthenticated) return;
    setIsLoading(true);
    try {
      const data = await appointmentsService.getMy();
      setAppointments(data);
    } catch (err) {
      console.error('Failed to load appointments', err);
    } finally {
      setIsLoading(false);
    }
  }, [isAuthenticated]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <AppointmentsContext.Provider value={{ appointments, isLoading, refresh }}>
      {children}
    </AppointmentsContext.Provider>
  );
}

export const useAppointments = () => useContext(AppointmentsContext);
