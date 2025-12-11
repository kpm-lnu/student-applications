import { Button, Dialog, DialogActions, DialogContent, DialogTitle } from '@mui/material'
import { useTranslation } from "react-i18next";

export const ExitDialog = ({
  isDialogOpened,
  onClose
}: {
  isDialogOpened: boolean
  onClose: () => void
}): React.JSX.Element => {
  const { t } = useTranslation()

  const handleNoClick = (): void => {
    onClose()
  }
  const handleYesClick = (): void => {
    window.electron.ipcRenderer.send('window-close')
  }

  return (
    <Dialog open={isDialogOpened}>
      <DialogTitle> {t('exit_dialog.title', 'Application Exit')}</DialogTitle>
      <DialogContent> {t('exit_dialog.text', 'Do you wand to exit the application?')}</DialogContent>
      <DialogActions>
        <Button variant="contained" onClick={() => handleYesClick()}>
          {t('exit_dialog.yes_button','Yes')}
        </Button>
        <Button variant="contained" onClick={() => handleNoClick()}>
          {t('exit_dialog.no_button','No')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
