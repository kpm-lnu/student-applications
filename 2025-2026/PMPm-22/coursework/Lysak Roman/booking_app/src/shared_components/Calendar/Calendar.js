import { useState, useEffect } from 'react';
import {
  CalendarWrapper,
  CalendarHeader,
  MonthYear,
  NavButton,
  WeekDays,
  WeekDay,
  DaysGrid,
  DayCell,
} from './Calendar_Styled';

const Calendar = ({ selectedDate, onDateSelect, minDate, maxDate }) => {
  const [currentMonth, setCurrentMonth] = useState(new Date());

  useEffect(() => {
    if (selectedDate) {
      setCurrentMonth(new Date(selectedDate));
    }
  }, [selectedDate]);

  const monthNames = [
    'Січень', 'Лютий', 'Березень', 'Квітень', 'Травень', 'Червень',
    'Липень', 'Серпень', 'Вересень', 'Жовтень', 'Листопад', 'Грудень'
  ];

  const weekDays = ['Нд', 'Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб'];

  const getDaysInMonth = (date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();

    const days = [];

    const prevMonthLastDay = new Date(year, month, 0).getDate();
    for (let i = startingDayOfWeek - 1; i >= 0; i--) {
      days.push({
        day: prevMonthLastDay - i,
        date: new Date(year, month - 1, prevMonthLastDay - i),
        isCurrentMonth: false,
      });
    }

    for (let day = 1; day <= daysInMonth; day++) {
      days.push({
        day,
        date: new Date(year, month, day),
        isCurrentMonth: true,
      });
    }

    const remainingDays = 42 - days.length; 
    for (let day = 1; day <= remainingDays; day++) {
      days.push({
        day,
        date: new Date(year, month + 1, day),
        isCurrentMonth: false,
      });
    }

    return days;
  };

  const isSameDay = (date1, date2) => {
    if (!date1 || !date2) return false;
    return (
      date1.getDate() === date2.getDate() &&
      date1.getMonth() === date2.getMonth() &&
      date1.getFullYear() === date2.getFullYear()
    );
  };

  const isToday = (date) => {
    return isSameDay(date, new Date());
  };

  const isDisabled = (date) => {
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    if (isSameDay(date, today)) return true;

    if (date < today) return true;

    const dayOfWeek = date.getDay();
    if (dayOfWeek === 0 || dayOfWeek === 6) return true;

    if (minDate) {
      const min = new Date(minDate);
      min.setHours(0, 0, 0, 0);
      if (date < min) return true;
    }

    if (maxDate) {
      const max = new Date(maxDate);
      max.setHours(0, 0, 0, 0);
      if (date > max) return true;
    }

    return false;
  };

  const handlePrevMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1));
  };

  const handleNextMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1));
  };

  const handleDayClick = (dayObj) => {
    if (!isDisabled(dayObj.date) && dayObj.isCurrentMonth) {
      onDateSelect(dayObj.date);
    }
  };

  const days = getDaysInMonth(currentMonth);

  const isPrevMonthDisabled = () => {
    const prevMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1);
    const today = new Date();
    return prevMonth.getMonth() < today.getMonth() && prevMonth.getFullYear() <= today.getFullYear();
  };

  return (
    <CalendarWrapper>
      <CalendarHeader>
        <NavButton type="button" onClick={handlePrevMonth} disabled={isPrevMonthDisabled()}>
          ←
        </NavButton>
        <MonthYear>
          {monthNames[currentMonth.getMonth()]} {currentMonth.getFullYear()}
        </MonthYear>
        <NavButton type="button" onClick={handleNextMonth}>
          →
        </NavButton>
      </CalendarHeader>

      <WeekDays>
        {weekDays.map((day) => (
          <WeekDay key={day}>{day}</WeekDay>
        ))}
      </WeekDays>

      <DaysGrid>
        {days.map((dayObj, index) => (
          <DayCell
            key={index}
            type="button"
            onClick={() => handleDayClick(dayObj)}
            $isSelected={selectedDate && isSameDay(dayObj.date, selectedDate)}
            $isToday={isToday(dayObj.date)}
            $isDisabled={isDisabled(dayObj.date)}
            $isOtherMonth={!dayObj.isCurrentMonth}
            disabled={isDisabled(dayObj.date)}
          >
            {dayObj.day}
          </DayCell>
        ))}
      </DaysGrid>
    </CalendarWrapper>
  );
};

export default Calendar;
