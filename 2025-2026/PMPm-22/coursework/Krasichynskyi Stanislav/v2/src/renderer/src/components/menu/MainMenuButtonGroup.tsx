import { Button, Paper } from '@mui/material'
import { ExitDialog } from '@renderer/components/menu/ExitDialog'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useTranslation } from 'react-i18next'

export const MainMenuButtonGroup = (): React.JSX.Element => {
  const navigate = useNavigate()
  const { t } = useTranslation()

  const [isExitDialogOpen, setIsExitDialogOpen] = useState(false)

  const handleNewSimulationClick = (): void => {

    //TODO. Replace
    // SimulationService.getInstance().reset()
    navigate('/new-simulation')
  }

  const handleLoadSimulationClick = (): void => {
    navigate('/saved-simulations')
  }

  const handleDocumentationClick = (): void => {
    window.electron.ipcRenderer.send('open-external')
  }

  const handleSettingsClick = (): void => {
    navigate('/settings')
  }

  const handleExitClick = (): void => {
    setIsExitDialogOpen(!isExitDialogOpen)
  }

  return (
    <Paper
      style={{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between'
      }}
      elevation={3}
    >
      <Button
        size="medium"
        variant="contained"
        sx={{ m: 1, flex: 1 }}
        onClick={() => handleNewSimulationClick()}
      >
        {t('home.new_simulation_button', 'New Simulation')}
      </Button>
      <Button
        size="medium"
        variant="contained"
        sx={{ m: 1, flex: 1 }}
        onClick={() => handleLoadSimulationClick()}
      >
        {t('home.load_simulation_button', 'Load Simulation')}

      </Button>
      <Button
        size="medium"
        variant="contained"
        sx={{ m: 1, flex: 1 }}
        onClick={() => handleDocumentationClick()}
      >
        {t('home.documentation_button', 'Documentation')}
      </Button>
      <Button
        size="medium"
        variant="contained"
        sx={{ m: 1, flex: 1 }}
        onClick={() => handleSettingsClick()}
      >
        {t('home.settings_button', 'Settings')}
      </Button>
      <Button
        size="medium"
        variant="contained"
        sx={{ m: 1, flex: 1 }}
        onClick={() => handleExitClick()}
      >
        {t('home.exit_button', 'Exit')}
      </Button>
      <ExitDialog isDialogOpened={isExitDialogOpen} onClose={handleExitClick} />
    </Paper>
  )
}
