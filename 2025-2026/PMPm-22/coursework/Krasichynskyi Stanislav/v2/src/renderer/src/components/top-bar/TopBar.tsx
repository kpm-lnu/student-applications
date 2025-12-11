import { IconButton, Paper } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'
import MinimizeIcon from '@mui/icons-material/Minimize'
import CropSquareIcon from '@mui/icons-material/CropSquare'

export const TopBar = (): React.JSX.Element => (
  <Paper
    square
    sx={{
      top: 0,
      left: 0,
      right: 0,
      height: '2rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'flex-start',
      WebkitAppRegion: 'drag',
      zIndex: 1000
    }}
  >
    <IconButton
      size="small"
      sx={{ WebkitAppRegion: 'no-drag', color: 'red' }}
      onClick={() => window.electron.ipcRenderer.send('window-close')}
    >
      <CloseIcon fontSize="small" />
    </IconButton>
    <IconButton
      size="small"
      sx={{ WebkitAppRegion: 'no-drag', color: 'yellow' }}
      onClick={() => window.electron.ipcRenderer.send('window-minimize')}
    >
      <MinimizeIcon fontSize="small" />
    </IconButton>
    <IconButton
      size="small"
      sx={{ WebkitAppRegion: 'no-drag', color: 'lawngreen' }}
      onClick={() => window.electron.ipcRenderer.send('window-maximize')}
    >
      <CropSquareIcon fontSize="small" />
    </IconButton>
  </Paper>
)
