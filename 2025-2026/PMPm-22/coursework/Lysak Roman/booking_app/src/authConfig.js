const CLIENT_ID = process.env.REACT_APP_CLIENT_ID;
const TENANT_ID = process.env.REACT_APP_TENANT_ID;

export const msalConfig = {
  auth: {
    clientId: CLIENT_ID,
    authority: `https://login.microsoftonline.com/${TENANT_ID}`,
    redirectUri: "http://localhost:3000",
  },
  cache: {
    cacheLocation: "sessionStorage",
    storeAuthStateInCookie: false,
  },
};

export const loginRequest = {
  scopes: [
    "User.Read",
    "BookingsAppointment.ReadWrite.All",
    "Bookings.Read.All"
  ],
};