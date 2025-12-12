import { useState, useEffect } from 'react';
import {
  TimeSlotWrapper,
  TimeSlotsGrid,
  TimeSlotButton,
  NoSlotsMessage,
} from './TimeSlotPicker_Styled';

const TimeSlotPicker = ({
  selectedDate,
  selectedTime,
  onTimeSelect,
}) => {
  const [availableSlots, setAvailableSlots] = useState([]);

  useEffect(() => {
    if (selectedDate) {
      generateTimeSlots();
    }
  }, [selectedDate]);

  const generateTimeSlots = () => {
    const slots = [];
    for (let hour = 9; hour < 21; hour++) {
      for (let minute = 0; minute < 60; minute += 30) {
        const slotDateTime = new Date(selectedDate);
        slotDateTime.setHours(hour, minute, 0, 0);

        slots.push({
          time: formatTimeDisplay(hour, minute),
          dateTime: slotDateTime,
        });
      }
    }

    const finalSlot = new Date(selectedDate);
    finalSlot.setHours(21, 0, 0, 0);
    slots.push({
      time: formatTimeDisplay(21, 0),
      dateTime: finalSlot,
    });

    setAvailableSlots(slots);
  };

  const formatTimeDisplay = (hours, minutes) => {
    const period = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes.toString().padStart(2, '0');
    return `${displayHours}:${displayMinutes} ${period}`;
  };

  if (!selectedDate) {
    return <NoSlotsMessage>Будь ласка, спочатку оберіть дату</NoSlotsMessage>;
  }

  if (availableSlots.length === 0) {
    return <NoSlotsMessage>Немає доступних часових слотів для цієї дати</NoSlotsMessage>;
  }

  return (
    <TimeSlotWrapper>
      <TimeSlotsGrid>
        {availableSlots.map((slot, index) => (
          <TimeSlotButton
            key={index}
            type="button"
            onClick={() => onTimeSelect(slot.dateTime)}
            $isSelected={selectedTime && slot.dateTime.getTime() === selectedTime.getTime()}
          >
            {slot.time}
          </TimeSlotButton>
        ))}
      </TimeSlotsGrid>
    </TimeSlotWrapper>
  );
};

export default TimeSlotPicker;
