import { StyledButton } from "./Button_Styled";

const Button = ({ children, ...props }) => (
    <StyledButton {...props}>{children}</StyledButton>
);

export default Button;