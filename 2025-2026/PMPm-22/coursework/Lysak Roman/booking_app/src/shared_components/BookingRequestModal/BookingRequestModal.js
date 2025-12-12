import {
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalTitle,
  CloseButton,
  BookingInfo,
  BookingInfoTitle,
  BookingInfoText,
} from './BookingRequestModal_Styled';

const BookingRequestModal = ({ isOpen, onClose, booking, service, staffMembers }) => {
  if (!isOpen) return null;

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

  const getAssignedStaff = () => {
    const serviceStaffIds = service.staffMemberIds || [];
    if (serviceStaffIds.length === 0) return 'Співробітників не призначено';

    const assignedStaff = staffMembers.filter(staff =>
      serviceStaffIds.includes(staff.id)
    );

    if (assignedStaff.length === 0) return 'Співробітників не призначено';

    return assignedStaff.map(staff => staff.displayName).join(', ');
  };

  const handleOverlayClick = (e) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <ModalOverlay onClick={handleOverlayClick}>
      <ModalContent>
        <ModalHeader>
          <ModalTitle>Інформація про послугу</ModalTitle>
          <CloseButton onClick={onClose} type="button">
            &times;
          </CloseButton>
        </ModalHeader>

        <BookingInfo>
          <BookingInfoTitle>{booking.displayName}</BookingInfoTitle>
          {booking.description && (
            <BookingInfoText>{booking.description}</BookingInfoText>
          )}
          {booking.phone && (
            <BookingInfoText><strong>Телефон:</strong> {booking.phone}</BookingInfoText>
          )}
          {booking.email && (
            <BookingInfoText><strong>Email:</strong> {booking.email}</BookingInfoText>
          )}
          {booking.webSiteUrl && (
            <BookingInfoText>
              <strong>Веб-сайт:</strong>{' '}
              <a href={booking.webSiteUrl} target="_blank" rel="noopener noreferrer">
                {booking.webSiteUrl}
              </a>
            </BookingInfoText>
          )}
        </BookingInfo>

        <BookingInfo>
          <BookingInfoTitle>Послуга: {service.displayName}</BookingInfoTitle>
          {service.description && (
            <BookingInfoText>{service.description}</BookingInfoText>
          )}
          <BookingInfoText>
            {service.defaultDuration && (
              <div style={{ marginBottom: '8px' }}>
                <strong>Тривалість:</strong> {formatDuration(service.defaultDuration)}
              </div>
            )}
            {service.defaultPrice !== undefined && (
              <div style={{ marginBottom: '8px' }}>
                <strong>Ціна:</strong> {formatPrice(service.defaultPrice)}
              </div>
            )}
            {service.defaultPriceType && (
              <div style={{ marginBottom: '8px' }}>
                <strong>Тип ціни:</strong> {service.defaultPriceType}
              </div>
            )}
            <div>
              <strong>Призначені співробітники:</strong> {getAssignedStaff()}
            </div>
          </BookingInfoText>
        </BookingInfo>

        <BookingInfo style={{ textAlign: 'center', background: 'rgba(20, 27, 77, 0.1)' }}>
          <BookingInfoText>
            {`Щоб забронювати зустріч для цієї послуги, будь ласка, використайте кнопку "${booking.displayName}"
            у списку бронювань.`}
          </BookingInfoText>
        </BookingInfo>
      </ModalContent>
    </ModalOverlay>
  );
};

export default BookingRequestModal;
