import { createContext, useState, useContext, useEffect } from 'react';
import { useMsal } from '@azure/msal-react';
import axios from 'axios';
import { loginRequest } from '../authConfig';
import { ALLOWED_BOOKINGS } from '../allowedBookings';

const BookingsContext = createContext();

export const useBookings = () => {
  const context = useContext(BookingsContext);
  if (!context) {
    throw new Error('useBookings must be used within BookingsProvider');
  }
  return context;
};

export const BookingsProvider = ({ children }) => {
  const { instance, accounts } = useMsal();
  const [bookings, setBookings] = useState([]);
  const [appointments, setAppointments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchBookings = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      const accessToken = response.accessToken;

      const bookingsResponse = await axios.get(
        'https://graph.microsoft.com/v1.0/solutions/bookingBusinesses',
        {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );
      const allBookings = bookingsResponse.data.value;

      const filtered = allBookings.filter((booking) =>
        Object.keys(ALLOWED_BOOKINGS).includes(booking.id)
      );

      const detailedBookings = await Promise.all(
        filtered.map(async (booking) => {
          const detailResponse = await axios.get(
            `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}`,
            {
              headers: {
                Authorization: `Bearer ${accessToken}`,
              },
            }
          );
          const bookingDetails = detailResponse.data;

          let services = [];
          try {
            const servicesResponse = await axios.get(
              `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}/services`,
              {
                headers: {
                  Authorization: `Bearer ${accessToken}`,
                },
              }
            );
            services = servicesResponse.data.value || [];
          } catch (error) {
            console.error(`Error fetching services for ${booking.id}:`, error);
          }

          let staffMembers = [];
          try {
            const staffResponse = await axios.get(
              `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}/staffMembers`,
              {
                headers: {
                  Authorization: `Bearer ${accessToken}`,
                },
              }
            );
            staffMembers = staffResponse.data.value || [];
          } catch (error) {
            console.error(`Error fetching staff members for ${booking.id}:`, error);
          }

          return { ...bookingDetails, services, staffMembers };
        })
      );

      setBookings(detailedBookings);
    } catch (error) {
      console.error('Error fetching bookings:', error);
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const getServiceById = (serviceId) => {
    for (const booking of bookings) {
      const service = booking.services?.find((s) => s.id === serviceId);
      if (service) {
        return { service, booking };
      }
    }
    return null;
  };

  const getBookingById = (bookingId) => {
    return bookings.find((b) => b.id === bookingId);
  };

  const getAllServices = () => {
    const allServices = [];
    bookings.forEach((booking) => {
      booking.services?.forEach((service) => {
        allServices.push({
          ...service,
          bookingId: booking.id,
          bookingName: booking.displayName,
          publicUrl: booking.publicUrl,
        });
      });
    });
    return allServices;
  };

  const fetchUserAppointments = async () => {
    if (!instance || !accounts || accounts.length === 0 || bookings.length === 0) {
      return;
    }

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      const accessToken = response.accessToken;
      const userEmail = accounts[0].username;

      const allAppointments = await Promise.all(
        bookings.map(async (booking) => {
          try {
            const appointmentsResponse = await axios.get(
              `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}/appointments`,
              {
                headers: {
                  Authorization: `Bearer ${accessToken}`,
                },
              }
            );

            const userAppointments = (appointmentsResponse.data.value || [])
              .filter(appt => appt.customerEmailAddress?.toLowerCase() === userEmail.toLowerCase())
              .map(appt => ({
                ...appt,
                bookingName: booking.displayName,
                bookingId: booking.id,
                serviceName: booking.services?.find(s => s.id === appt.serviceId)?.displayName || 'Невідома послуга',
              }));

            return userAppointments;
          } catch (error) {
            console.error(`Error fetching appointments for ${booking.id}:`, error);
            return [];
          }
        })
      );

      const flatAppointments = allAppointments.flat();
      flatAppointments.sort((a, b) => {
        const dateA = new Date(a.startDateTime?.dateTime || 0);
        const dateB = new Date(b.startDateTime?.dateTime || 0);
        return dateA - dateB;
      });

      setAppointments(flatAppointments);
    } catch (error) {
      console.error('Error fetching user appointments:', error);
    }
  };

  const deleteAppointment = async (appointmentId, bookingId) => {
    if (!instance || !accounts || accounts.length === 0) {
      throw new Error('Не авторизовано');
    }

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      const accessToken = response.accessToken;

      await axios.delete(
        `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${bookingId}/appointments/${appointmentId}`,
        {
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );

      await fetchUserAppointments();
      return { success: true };
    } catch (error) {
      console.error('Error deleting appointment:', error);
      throw new Error(error.response?.data?.error?.message || 'Не вдалося видалити зустріч');
    }
  };

  useEffect(() => {
    if (accounts && accounts.length > 0) {
      fetchBookings();
    }
  }, [instance, accounts]);

  useEffect(() => {
    if (bookings.length > 0 && accounts && accounts.length > 0) {
      fetchUserAppointments();
    }
  }, [bookings, instance, accounts]);

  const value = {
    bookings,
    appointments,
    loading,
    error,
    fetchBookings,
    fetchUserAppointments,
    deleteAppointment,
    getServiceById,
    getBookingById,
    getAllServices,
  };

  return (
    <BookingsContext.Provider value={value}>
      {children}
    </BookingsContext.Provider>
  );
};
