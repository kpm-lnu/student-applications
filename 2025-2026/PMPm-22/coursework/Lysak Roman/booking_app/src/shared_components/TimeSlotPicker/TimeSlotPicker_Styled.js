import styled from 'styled-components';

export const TimeSlotWrapper = styled.div`
  width: 100%;
`;

export const TimeSlotsGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  max-height: 200px;
  overflow-y: auto;
  padding: 4px;
`;

export const TimeSlotButton = styled.button`
  padding: 10px;
  border: 2px solid ${props =>
    props.$isSelected
      ? 'var(--main-bg-color, #141B4D)'
      : '#e0e0e0'
  };
  background: ${props =>
    props.$isSelected
      ? 'var(--main-bg-color, #141B4D)'
      : 'white'
  };
  color: ${props =>
    props.$isSelected
      ? 'white'
      : 'var(--main-black-text-color, #000)'
  };
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: ${props => props.$isSelected ? '600' : '400'};
  transition: all 0.2s;

  &:hover {
    background: ${props =>
      props.$isSelected
        ? 'var(--accent-color, #0f1536)'
        : 'rgba(20, 27, 77, 0.05)'
    };
  }

  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
  }
`;

export const LoadingMessage = styled.div`
  text-align: center;
  padding: 20px;
  color: #666;
  font-size: 14px;
`;

export const NoSlotsMessage = styled.div`
  text-align: center;
  padding: 20px;
  color: #999;
  font-size: 14px;
`;
