import { useState } from "react";
import { useBookings } from "../../contexts/BookingsContext";
import {
  Wrapper,
  List,
  ListItem,
  ServicesDropdown,
  DropdownHeader,
  DropdownArrow,
  DropdownContent,
  ServiceItem,
  ServiceName,
  ServiceDescription,
  ServiceDetails
} from './Bookings_Styled';
import Button from "../../shared_components/Button/Button";
import BookingRequestModal from "../../shared_components/BookingRequestModal/BookingRequestModal";
import AppointmentsModal from "../../shared_components/AppointmentsModal/AppointmentsModal";

function Bookings() {
  const { bookings, loading } = useBookings();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isAppointmentsModalOpen, setIsAppointmentsModalOpen] = useState(false);
  const [selectedBooking, setSelectedBooking] = useState(null);
  const [selectedService, setSelectedService] = useState(null);
  const [openDropdownId, setOpenDropdownId] = useState(null);

  const handleServiceClick = (booking, service) => {
    setSelectedBooking(booking);
    setSelectedService(service);
    setIsModalOpen(true);
    setOpenDropdownId(null); 
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
    setSelectedBooking(null);
    setSelectedService(null);
  };

  const handleOpenAppointments = (booking) => {
    setSelectedBooking(booking);
    setIsAppointmentsModalOpen(true);
  };

  const handleCloseAppointments = () => {
    setIsAppointmentsModalOpen(false);
    setSelectedBooking(null);
  };

  const toggleDropdown = (bookingId) => {
    setOpenDropdownId(openDropdownId === bookingId ? null : bookingId);
  };

  const formatDuration = (duration) => {
    if (!duration) return 'Н/Д';
    const match = duration.match(/PT(\d+)H?(\d+)?M?/);
    if (!match) return duration;
    const hours = match[1] ? `${match[1]}год` : '';
    const minutes = match[2] ? `${match[2]}хв` : '';
    return `${hours} ${minutes}`.trim();
  };

  const formatPrice = (price) => {
    if (!price) return '';
    return `$${price}`;
  };

  if (loading) return <div style={{ paddingLeft: '20px', fontWeight: "bold", fontSize: '20px' }}>Завантаження бронювань...</div>;

  return (
    <Wrapper>
      <h2>Ваші бронювання</h2>
      <List role="list">
        {bookings.map((booking) => (
          <ListItem key={booking.id} role="listitem">
            <div style={{ display: 'flex', gap: '12px', marginBottom: '12px' }}>
              <Button onClick={() => window.open(`https://outlook.office.com/book/${booking.id}/`, "_blank")}>
                {booking.displayName}
              </Button>
              <Button onClick={() => handleOpenAppointments(booking)}>
                Управління зустрічами
              </Button>
            </div>
            <div>
              <strong>Телефон:</strong> {booking.phone || "Н/Д"}
            </div>
            <div>
              <strong>Email:</strong> {booking.email || "Н/Д"}
            </div>
            <div>
              <strong>Веб-сайт:</strong> {booking.webSiteUrl ? (
                <a href={booking.webSiteUrl} target="_blank" rel="noopener noreferrer">{booking.webSiteUrl}</a>
              ) : "Н/Д"}
            </div>
            <div>
              <strong>Адреса:</strong>{booking.address?.city}, {booking.address?.countryOrRegion}
            </div>
            <div>
              <strong>Тип бізнесу:</strong> {booking.businessType || "Н/Д"}
            </div>

            {booking.services && booking.services.length > 0 && (
              <ServicesDropdown>
                <DropdownHeader onClick={() => toggleDropdown(`services-${booking.id}`)}>
                  <span>Послуги ({booking.services.length})</span>
                  <DropdownArrow $isOpen={openDropdownId === `services-${booking.id}`}>
                    ▼
                  </DropdownArrow>
                </DropdownHeader>
                <DropdownContent $isOpen={openDropdownId === `services-${booking.id}`}>
                  {booking.services.map((service) => (
                    <ServiceItem
                      key={service.id}
                      onClick={() => handleServiceClick(booking, service)}
                    >
                      <ServiceName>{service.displayName}</ServiceName>
                      {service.description && (
                        <ServiceDescription>{service.description}</ServiceDescription>
                      )}
                      <ServiceDetails>
                        {service.defaultDuration && (
                          <span><strong>Тривалість:</strong> {formatDuration(service.defaultDuration)}</span>
                        )}
                        {service.defaultPrice !== undefined && (
                          <span><strong>Ціна:</strong> {formatPrice(service.defaultPrice)}</span>
                        )}
                      </ServiceDetails>
                    </ServiceItem>
                  ))}
                </DropdownContent>
              </ServicesDropdown>
            )}

            {booking.staffMembers && booking.staffMembers.length > 0 && (
              <ServicesDropdown>
                <DropdownHeader onClick={() => toggleDropdown(`staff-${booking.id}`)}>
                  <span>Співробітники ({booking.staffMembers.length})</span>
                  <DropdownArrow $isOpen={openDropdownId === `staff-${booking.id}`}>
                    ▼
                  </DropdownArrow>
                </DropdownHeader>
                <DropdownContent $isOpen={openDropdownId === `staff-${booking.id}`}>
                  {booking.staffMembers.map((staff) => (
                    <ServiceItem key={staff.id}>
                      <ServiceName>{staff.displayName}</ServiceName>
                      {staff.emailAddress && (
                        <ServiceDescription>Email: {staff.emailAddress}</ServiceDescription>
                      )}
                      <ServiceDetails>
                        {staff.role && (
                          <span><strong>Роль:</strong> {staff.role}</span>
                        )}
                      </ServiceDetails>
                    </ServiceItem>
                  ))}
                </DropdownContent>
              </ServicesDropdown>
            )}
          </ListItem>
        ))}
      </List>

      {selectedBooking && selectedService && (
        <BookingRequestModal
          isOpen={isModalOpen}
          onClose={handleCloseModal}
          booking={selectedBooking}
          service={selectedService}
          staffMembers={selectedBooking.staffMembers || []}
        />
      )}

      {isAppointmentsModalOpen && selectedBooking && (
        <AppointmentsModal
          isOpen={isAppointmentsModalOpen}
          onClose={handleCloseAppointments}
          booking={selectedBooking}
        />
      )}
    </Wrapper>
  );
}

export default Bookings;