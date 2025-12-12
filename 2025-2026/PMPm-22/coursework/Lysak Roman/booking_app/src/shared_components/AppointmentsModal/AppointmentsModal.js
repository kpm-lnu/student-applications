import { useState } from 'react';
import { useMsal } from '@azure/msal-react';
import axios from 'axios';
import { loginRequest } from '../../authConfig';
import { useBookings } from '../../contexts/BookingsContext';
import Calendar from '../Calendar/Calendar';
import TimeSlotPicker from '../TimeSlotPicker/TimeSlotPicker';
import {
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalTitle,
  CloseButton,
  AppointmentsList,
  AppointmentCard,
  AppointmentHeader,
  AppointmentTitle,
  AppointmentInfo,
  AppointmentActions,
  ActionButton,
  AddButton,
  Form,
  FormGroup,
  Label,
  Input,
  Select,
  ButtonGroup,
  SubmitButton,
  CancelButton,
  EmptyState,
} from './AppointmentsModal_Styled';

const AppointmentsModal = ({ isOpen, onClose, booking }) => {
  const { instance, accounts } = useMsal();
  const { appointments: allAppointments, loading, deleteAppointment, fetchUserAppointments } = useBookings();
  const [isFormOpen, setIsFormOpen] = useState(false);
  const [editingAppointment, setEditingAppointment] = useState(null);
  const appointments = allAppointments.filter(appt => appt.bookingId === booking?.id);

  const getUserData = () => {
    const userEmail = accounts && accounts.length > 0 ? accounts[0].username : localStorage.getItem('userEmail') || '';
    const userName = accounts && accounts.length > 0 ? accounts[0].name : localStorage.getItem('userName') || '';

    if (userEmail) localStorage.setItem('userEmail', userEmail);
    if (userName) localStorage.setItem('userName', userName);

    return { userEmail, userName };
  };

  const [formData, setFormData] = useState({
    customerName: getUserData().userName,
    customerEmailAddress: getUserData().userEmail,
    customerPhone: '',
    serviceId: '',
    staffMemberId: '',
  });
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedTime, setSelectedTime] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleAddNew = () => {
    const { userEmail, userName } = getUserData();
    setEditingAppointment(null);
    setFormData({
      customerName: userName,
      customerEmailAddress: userEmail,
      customerPhone: '',
      serviceId: booking.services?.[0]?.id || '',
      staffMemberId: '',
    });
    setSelectedDate(null);
    setSelectedTime(null);
    setIsFormOpen(true);
  };

  const handleEdit = (appointment) => {
    setEditingAppointment(appointment);
    setFormData({
      customerName: appointment.customerName || '',
      customerEmailAddress: appointment.customerEmailAddress || '',
      customerPhone: appointment.customerPhone || '',
      serviceId: appointment.serviceId || '',
      staffMemberId: appointment.staffMemberIds?.[0] || '',
    });

    if (appointment.startDateTime?.dateTime) {
      const appointmentDateTime = new Date(appointment.startDateTime.dateTime);
      setSelectedDate(appointmentDateTime);
      setSelectedTime(appointmentDateTime);
    } else {
      setSelectedDate(null);
      setSelectedTime(null);
    }

    setIsFormOpen(true);
  };

  const handleCancelForm = () => {
    const { userEmail, userName } = getUserData();
    setIsFormOpen(false);
    setEditingAppointment(null);
    setFormData({
      customerName: userName,
      customerEmailAddress: userEmail,
      customerPhone: '',
      serviceId: '',
      staffMemberId: '',
    });
    setSelectedDate(null);
    setSelectedTime(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!selectedDate || !selectedTime) {
      alert('Будь ласка, оберіть дату та час для зустрічі.');
      return;
    }

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: accounts[0],
      });
      const accessToken = response.accessToken;

      const selectedService = booking.services?.find(s => s.id === formData.serviceId);
      const serviceDuration = parseDuration(selectedService?.defaultDuration || 'PT1H');
      const startTime = new Date(selectedTime);
      const endTime = new Date(startTime.getTime() + serviceDuration * 60000);

      const formatDateTime = (date) => {
        return date.toISOString();
      };

      const appointmentData = {
        "@odata.type": "#microsoft.graph.bookingAppointment",
        additionalInformation: "",
        isLocationOnline: false,
        joinWebUrl: "",
        customerName: formData.customerName,
        customerEmailAddress: formData.customerEmailAddress,
        customerTimeZone: "UTC",
        customerNotes: "",
        serviceId: formData.serviceId,
        serviceName: selectedService?.displayName || '',
        duration: selectedService?.defaultDuration || 'PT1H',
        preBuffer: "PT0S",
        postBuffer: "PT0S",
        priceType: selectedService?.priceType || "fixedPrice",
        price: selectedService?.defaultPrice || 0,
        serviceNotes: "",
        optOutOfCustomerEmail: false,
        smsNotificationsEnabled: false,
        anonymousJoinWebUrl: "",
        maximumAttendeesCount: 1,
        filledAttendeesCount: 1,
        isCustomerAllowedToManageBooking: true,
        appointmentLabel: "",
        startDateTime: {
          dateTime: formatDateTime(startTime),
          timeZone: "UTC"
        },
        endDateTime: {
          dateTime: formatDateTime(endTime),
          timeZone: "UTC"
        }
      };

      if (formData.customerPhone && formData.customerPhone.trim()) {
        appointmentData.customerPhone = formData.customerPhone.trim();
      }

      if (formData.staffMemberId) {
        appointmentData.staffMemberIds = [formData.staffMemberId];
      }

      console.log('Sending appointment data:', JSON.stringify(appointmentData, null, 2));

      if (editingAppointment) {
        await axios.patch(
          `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}/appointments/${editingAppointment.id}`,
          appointmentData,
          {
            headers: {
              Authorization: `Bearer ${accessToken}`,
              'Content-Type': 'application/json',
            },
          }
        );
        alert('Зустріч успішно оновлено!');
      } else {
        await axios.post(
          `https://graph.microsoft.com/v1.0/solutions/bookingBusinesses/${booking.id}/appointments`,
          appointmentData,
          {
            headers: {
              Authorization: `Bearer ${accessToken}`,
              'Content-Type': 'application/json',
            },
          }
        );
        alert('Зустріч успішно створено!');
      }

      handleCancelForm();
      await fetchUserAppointments();
    } catch (error) {
      console.error('Error saving appointment:', error);
      console.error('Error response:', error.response?.data);
      const errorMessage = error.response?.data?.error?.message || error.response?.data?.message || 'Не вдалося зберегти зустріч. Будь ласка, спробуйте ще раз.';
      alert(`Помилка: ${errorMessage}\n\nБудь ласка, перевірте консоль для більш детальної інформації.`);
    }
  };

  const handleDelete = async (appointmentId) => {
    if (!window.confirm('Ви впевнені, що хочете видалити цю зустріч?')) {
      return;
    }

    try {
      await deleteAppointment(appointmentId, booking.id);
      alert('Зустріч успішно видалено!');
    } catch (error) {
      console.error('Error deleting appointment:', error);
      alert(error.message || 'Не вдалося видалити зустріч. Будь ласка, спробуйте ще раз.');
    }
  };

  const parseDuration = (duration) => {
    if (!duration) return 60;
    const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
    if (!match) return 60;
    const hours = parseInt(match[1] || 0);
    const minutes = parseInt(match[2] || 0);
    return hours * 60 + minutes;
  };

  const formatDateTime = (dateTimeObj) => {
    if (!dateTimeObj?.dateTime) return 'Н/Д';
    return new Date(dateTimeObj.dateTime).toLocaleString('uk-UA');
  };

  const getServiceName = (serviceId) => {
    return booking.services?.find(s => s.id === serviceId)?.displayName || 'Невідома послуга';
  };

  const getStaffName = (staffIds) => {
    if (!staffIds || staffIds.length === 0) return 'Не призначено';
    const staff = booking.staffMembers?.find(s => s.id === staffIds[0]);
    return staff?.displayName || 'Невідомий співробітник';
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <ModalOverlay onClick={handleOverlayClick}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Управління зустрічами - {booking.displayName}</ModalTitle>
          <CloseButton onClick={onClose} type="button">
            &times;
          </CloseButton>
        </ModalHeader>

        {!isFormOpen && (
          <AddButton onClick={handleAddNew}>
            + Додати нову зустріч
          </AddButton>
        )}

        {isFormOpen && (
          <Form onSubmit={handleSubmit}>
            <h3>{editingAppointment ? 'Редагувати зустріч' : 'Нова зустріч'}</h3>

            <FormGroup>
              <Label htmlFor="serviceId">Послуга *</Label>
              <Select
                id="serviceId"
                name="serviceId"
                value={formData.serviceId}
                onChange={handleChange}
                required
              >
                <option value="">Оберіть послугу</option>
                {booking.services?.map((service) => (
                  <option key={service.id} value={service.id}>
                    {service.displayName}
                  </option>
                ))}
              </Select>
            </FormGroup>

            <FormGroup>
              <Label>Оберіть дату *</Label>
              <Calendar
                selectedDate={selectedDate}
                onDateSelect={setSelectedDate}
              />
            </FormGroup>

            {selectedDate && (
              <FormGroup>
                <Label>Оберіть час *</Label>
                <TimeSlotPicker
                  selectedDate={selectedDate}
                  selectedTime={selectedTime}
                  onTimeSelect={setSelectedTime}
                />
              </FormGroup>
            )}

            <FormGroup>
              <Label htmlFor="staffMemberId">Співробітник</Label>
              <Select
                id="staffMemberId"
                name="staffMemberId"
                value={formData.staffMemberId}
                onChange={handleChange}
                required
              >
                <option value="">Оберіть співробітника</option>
                {booking.staffMembers?.map((staff) => (
                  <option key={staff.id} value={staff.id}>
                    {staff.displayName}
                  </option>
                ))}
              </Select>
            </FormGroup>

            <FormGroup>
              <Label htmlFor="customerName">Ім'я клієнта *</Label>
              <Input
                type="text"
                id="customerName"
                name="customerName"
                value={formData.customerName}
                onChange={handleChange}
                required
                placeholder="Введіть ім'я клієнта"
              />
            </FormGroup>

            <FormGroup>
              <Label htmlFor="customerEmailAddress">Email клієнта *</Label>
              <Input
                type="email"
                id="customerEmailAddress"
                name="customerEmailAddress"
                value={formData.customerEmailAddress}
                onChange={handleChange}
                required
                placeholder="client@example.com"
              />
            </FormGroup>

            <FormGroup>
              <Label htmlFor="customerPhone">Телефон клієнта</Label>
              <Input
                type="tel"
                id="customerPhone"
                name="customerPhone"
                value={formData.customerPhone}
                onChange={handleChange}
                placeholder="+380 XX XXX XX XX (необов'язково)"
              />
            </FormGroup>

            <ButtonGroup>
              <CancelButton type="button" onClick={handleCancelForm}>
                Скасувати
              </CancelButton>
              <SubmitButton type="submit">
                {editingAppointment ? 'Оновити зустріч' : 'Створити зустріч'}
              </SubmitButton>
            </ButtonGroup>
          </Form>
        )}

        {loading ? (
          <EmptyState>Завантаження зустрічей...</EmptyState>
        ) : appointments.length === 0 ? (
          <EmptyState>Зустрічей не знайдено. Додайте вашу першу зустріч вище.</EmptyState>
        ) : (
          <AppointmentsList>
            {appointments.map((appointment) => (
              <AppointmentCard key={appointment.id}>
                <AppointmentHeader>
                  <AppointmentTitle>{appointment.customerName}</AppointmentTitle>
                </AppointmentHeader>
                <AppointmentInfo>
                  <div><strong>Послуга:</strong> {getServiceName(appointment.serviceId)}</div>
                  <div><strong>Початок:</strong> {formatDateTime(appointment.startDateTime)}</div>
                  <div><strong>Кінець:</strong> {formatDateTime(appointment.endDateTime)}</div>
                  <div><strong>Email:</strong> {appointment.customerEmailAddress}</div>
                  <div><strong>Телефон:</strong> {appointment.customerPhone}</div>
                  <div><strong>Співробітник:</strong> {getStaffName(appointment.staffMemberIds)}</div>
                  {appointment.price !== undefined && (
                    <div><strong>Ціна:</strong> ${appointment.price}</div>
                  )}
                </AppointmentInfo>
                <AppointmentActions>
                  <ActionButton onClick={() => handleEdit(appointment)}>
                    Редагувати
                  </ActionButton>
                  <ActionButton $danger onClick={() => handleDelete(appointment.id)}>
                    Видалити
                  </ActionButton>
                </AppointmentActions>
              </AppointmentCard>
            ))}
          </AppointmentsList>
        )}
      </ModalContent>
    </ModalOverlay>
  );
};

export default AppointmentsModal;
