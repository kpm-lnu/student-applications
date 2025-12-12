import styled from 'styled-components';

export const CalendarWrapper = styled.div`
  width: 100%;
`;

export const CalendarHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
`;

export const MonthYear = styled.div`
  font-weight: 600;
  font-size: 16px;
  color: var(--main-bg-color, #141B4D);
`;

export const NavButton = styled.button`
  background: transparent;
  border: 2px solid var(--main-bg-color, #141B4D);
  border-radius: 4px;
  padding: 4px 12px;
  cursor: pointer;
  font-weight: 600;
  color: var(--main-bg-color, #141B4D);
  transition: all 0.2s;

  &:hover:not(:disabled) {
    background: rgba(20, 27, 77, 0.1);
  }

  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
`;

export const WeekDays = styled.div`
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 4px;
  margin-bottom: 8px;
`;

export const WeekDay = styled.div`
  text-align: center;
  font-weight: 600;
  font-size: 12px;
  color: #666;
  padding: 8px 0;
`;

export const DaysGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(7, 1fr);
  gap: 4px;
`;

export const DayCell = styled.button`
  aspect-ratio: 1;
  border: 2px solid ${props => {
    if (props.$isSelected) return 'var(--main-bg-color, #141B4D)';
    if (props.$isToday) return '#4CAF50';
    return '#e0e0e0';
  }};
  background: ${props => {
    if (props.$isSelected) return 'var(--main-bg-color, #141B4D)';
    if (props.$isToday) return 'rgba(76, 175, 80, 0.1)';
    return 'white';
  }};
  color: ${props => {
    if (props.$isSelected) return 'white';
    if (props.$isDisabled) return '#ccc';
    if (props.$isOtherMonth) return '#999';
    return 'var(--main-black-text-color, #000)';
  }};
  border-radius: 4px;
  cursor: ${props => props.$isDisabled ? 'not-allowed' : 'pointer'};
  font-weight: ${props => props.$isSelected ? '600' : '400'};
  font-size: 14px;
  transition: all 0.2s;
  opacity: ${props => props.$isOtherMonth ? 0.5 : 1};

  &:hover:not(:disabled) {
    background: ${props =>
      props.$isSelected
        ? 'var(--accent-color, #0f1536)'
        : 'rgba(20, 27, 77, 0.05)'
    };
  }

  &:disabled {
    cursor: not-allowed;
    opacity: 0.3;
  }
`;
