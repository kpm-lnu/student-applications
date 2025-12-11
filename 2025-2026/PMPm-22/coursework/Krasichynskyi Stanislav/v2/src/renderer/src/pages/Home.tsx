import { Box, Typography } from '@mui/material'
import { useEffect, useMemo, useState } from 'react'
import Particles, { initParticlesEngine } from '@tsparticles/react'
import { loadSlim } from '@tsparticles/slim'
import { ISourceOptions } from '@tsparticles/engine'
import { backgroundTheme } from '@renderer/theme/backgroundTheme'
import { MainMenuButtonGroup } from '@renderer/components/menu/MainMenuButtonGroup'
import { useTranslation } from "react-i18next";

export const Home = (): React.JSX.Element => {
  const [ready, setReady] = useState(false)
  const { t } = useTranslation();

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine)
    }).then(() => {
      setReady(true)
    })
  }, [])

  const particlesOptions = useMemo<ISourceOptions>(() => backgroundTheme, [])

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%'
      }}
    >
      <Typography variant="h4" gutterBottom color="text.primary">
        {t('home.title', 'CRSolver')}
      </Typography>
      {ready && <Particles id="tsparticles" options={particlesOptions} />}
      <MainMenuButtonGroup />
    </Box>
  )
}
