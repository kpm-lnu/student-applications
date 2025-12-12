import styled from 'styled-components';

export const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
`;

export const ModalContent = styled.div`
  background: white;
  border-radius: 8px;
  padding: 32px;
  max-width: 500px;
  width: 90%;
  max-height: 90vh;
  overflow-y: auto;
  box-shadow: 0 4px 16px rgba(20, 27, 77, 0.2);
  position: relative;
`;

export const ModalHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
`;

export const ModalTitle = styled.h2`
  margin: 0;
  color: var(--main-bg-color, #141B4D);
  font-size: 24px;
`;

export const CloseButton = styled.button`
  background: transparent;
  border: none;
  font-size: 28px;
  cursor: pointer;
  color: var(--main-bg-color, #141B4D);
  padding: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;

  &:hover {
    background: rgba(20, 27, 77, 0.1);
    border-radius: 4px;
  }
`;

export const Form = styled.form`
  display: flex;
  flex-direction: column;
  gap: 20px;
`;

export const FormGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

export const Label = styled.label`
  font-weight: 600;
  color: var(--main-black-text-color, #000);
  font-size: 14px;
`;

export const Input = styled.input`
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 16px;
  font-family: var(--font-family, 'Century Gothic', Arial, sans-serif);

  &:focus {
    outline: none;
    border-color: var(--main-bg-color, #141B4D);
  }
`;

export const TextArea = styled.textarea`
  padding: 12px;
  border: 2px solid #e0e0e0;
  border-radius: 6px;
  font-size: 16px;
  font-family: var(--font-family, 'Century Gothic', Arial, sans-serif);
  min-height: 100px;
  resize: vertical;

  &:focus {
    outline: none;
    border-color: var(--main-bg-color, #141B4D);
  }
`;

export const ButtonGroup = styled.div`
  display: flex;
  gap: 12px;
  justify-content: flex-end;
  margin-top: 8px;
`;

export const SubmitButton = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  background-color: var(--main-bg-color, #141B4D);
  color: var(--main-white-text-color, #fff);
  transition: background 0.2s;

  &:hover {
    background-color: var(--accent-color, #0f1536);
  }

  &:disabled {
    background-color: #cccccc;
    cursor: not-allowed;
  }
`;

export const CancelButton = styled.button`
  padding: 12px 24px;
  border: 2px solid var(--main-bg-color, #141B4D);
  background: white;
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  color: var(--main-bg-color, #141B4D);
  transition: all 0.2s;

  &:hover {
    background-color: rgba(20, 27, 77, 0.05);
  }
`;

export const BookingInfo = styled.div`
  background: rgba(20, 27, 77, 0.05);
  padding: 16px;
  border-radius: 6px;
  margin-bottom: 4px;
`;

export const BookingInfoTitle = styled.div`
  font-weight: 600;
  color: var(--main-bg-color, #141B4D);
  margin-bottom: 8px;
`;

export const BookingInfoText = styled.div`
  color: var(--main-black-text-color, #000);
  font-size: 14px;
`;
