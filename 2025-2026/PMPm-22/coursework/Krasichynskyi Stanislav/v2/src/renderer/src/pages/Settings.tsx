import { Autocomplete, Box, Button, Paper, TextField, Typography } from "@mui/material";
import { useTranslation } from "react-i18next";
import i18n from "@renderer/i18n";
import { useState } from "react";
import { ApplicationSettings } from '../../../shared/settings/ApplicationSettings';
import { AVAILABLE_COLOR_THEMES } from "../../../shared/constants";
import { useNavigate } from "react-router-dom";

export const Settings = (): React.JSX.Element => {
  const { t } = useTranslation();
  const navigate =useNavigate();
  const [selectedLanguage, setSelectedLanguage] = useState<string>(ApplicationSettings.getInstance().currentLanguage);

  const languages = (i18n.options.supportedLngs as string[] || [])
    .filter(l => l !== 'cimode' && true);

  const langOptions = languages.map(code => ({
    code,
    label: t(`languages:${code}`, code)
  }));

  const handleLanguageChange = (_: any, option: { code: string; label: string } | null): void => {
    if (!option) return;
    setSelectedLanguage(option.code);
    ApplicationSettings.getInstance().currentLanguage = option.code;
    void i18n.changeLanguage(option.code);
  }

  const handleCancelClick = () => {
    navigate('/')
  }

  return (
    <Box
      sx={{
        width: '100%',
        height: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        p: 12
      }}
    >
      <Paper
        elevation={4}
        sx={{
          p: 4
        }}
      >
        <Typography variant="h4" gutterBottom color="text.primary">
          {t('application_settings.title', 'Application Settings')}
        </Typography>
        <Typography>{t('application_settings.language_title', 'Language')}: </Typography>
        <Autocomplete
          disablePortal
          options={langOptions}
          value={langOptions.find((o) => o.code === selectedLanguage) ?? null}
          onChange={handleLanguageChange}
          renderInput={(params) => <TextField {...params} label="" />}
        />
        <Typography sx={{ mt: 2 }}>{t('application_settings.color_theme_title', 'Color Theme')}: </Typography>
        <Autocomplete
          disablePortal
          value={'dark'}
          options={AVAILABLE_COLOR_THEMES}
          renderInput={(params) => <TextField {...params} label="" />}
        />
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3, gap: 1 }}>
          <Button onClick={handleCancelClick}>{t('application_settings.cancel_button', 'Cancel')}</Button>
          <Button variant="contained">{t('application_settings.save_button', 'Save')}</Button>
        </Box>
      </Paper>
    </Box>
  )
}
