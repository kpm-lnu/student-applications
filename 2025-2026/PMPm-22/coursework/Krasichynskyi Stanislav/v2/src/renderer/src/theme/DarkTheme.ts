import { createTheme, Theme } from '@mui/material/styles'

export const darkTheme: Theme = createTheme({
  palette: {
    mode: 'dark',

    primary: {
      main: '#1DCD9F',
      contrastText: '#000'
    },

    secondary: {
      main: '#169976',
      contrastText: '#fff'
    },

    background: {
      default: '#222222',
      paper: '#2A2A2A'
    },

    text: {
      primary: '#ECECEC',
      secondary: '#7A7A7A'
    },

    error: { main: '#FF3366' },
    warning: { main: '#FFCC00' },
    info: { main: '#66FFFF' },
    success: { main: '#33FF77' }
  },

  typography: {
    fontFamily: '"Exo 2", sans-serif',
    button: { textTransform: 'none' }
  },

  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#1DCD9F'
        }
      }
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 0
        }
      }
    }
  }
})

export default darkTheme
