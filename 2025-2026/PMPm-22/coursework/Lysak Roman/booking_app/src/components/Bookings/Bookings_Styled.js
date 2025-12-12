import styled from "styled-components";

export const Wrapper = styled.div`
    padding: 0 20px;
    display: flex;
    flex-direction: column;
    gap: 5px;`

export const List = styled.div`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;`

export const ListItem = styled.div`
  border: 4px solid var(--main-bg-color, #141B4D);
  color: var(--main-black-text-color);
  font-family: var(--font-family, 'Century Gothic', Arial, sans-serif);
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(20, 27, 77, 0.1);
  padding: 24px;
  min-height: 150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  gap: 5px;
`

export const ServicesDropdown = styled.div`
  width: 100%;
  margin-top: 8px;
`

export const DropdownHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 12px;
  background: rgba(20, 27, 77, 0.05);
  border: 2px solid var(--main-bg-color, #141B4D);
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
  transition: all 0.2s;

  &:hover {
    background: rgba(20, 27, 77, 0.1);
  }
`

export const DropdownArrow = styled.span`
  font-size: 12px;
  transition: transform 0.2s;
  transform: ${props => props.$isOpen ? 'rotate(180deg)' : 'rotate(0deg)'};
`

export const DropdownContent = styled.div`
  max-height: ${props => props.$isOpen ? '300px' : '0'};
  overflow: hidden;
  transition: max-height 0.3s ease-in-out;
  border: ${props => props.$isOpen ? '2px solid var(--main-bg-color, #141B4D)' : 'none'};
  border-top: none;
  border-radius: 0 0 6px 6px;
  margin-top: -6px;
`

export const ServiceItem = styled.div`
  padding: 12px;
  border-bottom: 1px solid #e0e0e0;
  cursor: pointer;
  transition: all 0.2s;

  &:last-child {
    border-bottom: none;
  }

  &:hover {
    background: rgba(20, 27, 77, 0.1);
    transform: translateX(4px);
  }
`

export const ServiceName = styled.div`
  font-weight: 600;
  color: var(--main-bg-color, #141B4D);
  margin-bottom: 4px;
`

export const ServiceDescription = styled.div`
  font-size: 14px;
  color: #666;
  margin-bottom: 4px;
`

export const ServiceDetails = styled.div`
  font-size: 12px;
  color: #888;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
`
