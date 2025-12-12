import { useMsal, useIsAuthenticated } from "@azure/msal-react";
import { loginRequest } from "../authConfig";
import Bookings from "./Bookings/Bookings";
import ChatWidget from "./ChatWidget/ChatWidget";
import logo from "../static/logo.png";

import { Wrapper, Header, UserInfoWrapper, RightBlock, LeftBlock } from './App_Styled';

import Button from "../shared_components/Button/Button";

function App() {
  const { instance, accounts } = useMsal();
  const isAuthenticated = useIsAuthenticated();

  const userEmail = accounts && accounts.length > 0 ? accounts[0].username : null;
  const userName = accounts && accounts.length > 0 ? accounts[0].name : null;

  const handleLogin = () => {
    instance.loginRedirect(loginRequest);
  };

  const handleLogout = () => {
    instance.logoutRedirect();
  };

  return (
    <Wrapper>
      <Header>
        <LeftBlock>
          <div className="img-wrapper">
            <img src={logo} />
          </div>
          <h1>Веб-додаток бронювань</h1>
        </LeftBlock>
        <RightBlock>
          {isAuthenticated && userEmail && (
            <UserInfoWrapper>
              <span style={{ marginRight: "1em" }}>Ви увійшли як: {userEmail}</span>
              <span style={{ marginRight: "1em" }}>{userName}</span>
            </UserInfoWrapper>
          )}
          {!isAuthenticated ? (
            <Button onClick={handleLogin}>Увійти</Button>
          ) : (
            <Button onClick={handleLogout}>Вийти</Button>
          )}
        </RightBlock>
      </Header>

      {isAuthenticated && (
        <>
          <Bookings />
          <ChatWidget />
        </>
      )}
    </Wrapper>
  );
}

export default App;