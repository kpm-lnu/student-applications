import { StrictMode, useState, useMemo, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import { ThemeProvider, CssBaseline } from '@mui/material'
import { themes, ThemeName } from './theme'
import App from './App'
import './i18n'

export function Root(): ReactNode {
  const [mode] = useState<ThemeName>('dark')

  const theme = useMemo(() => themes[mode], [mode])

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <HashRouter>
        <App />
      </HashRouter>
    </ThemeProvider>
  )
}

const container = document.getElementById('root')!
const root = createRoot(container)

root.render(
  <StrictMode>
    <Root />
  </StrictMode>
)
