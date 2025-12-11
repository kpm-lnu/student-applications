import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import LanguageDetector from 'i18next-browser-languagedetector'

import en from './locales/en/translation.json'
import uk from './locales/uk/translation.json'

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    fallbackLng: 'en',
    supportedLngs: ['en', 'uk'],
    resources: {
      en: { translation: en },
      uk: { translation: uk }
    },
    interpolation: { escapeValue: false },
    detection: {
      order: ['querystring', 'cookie', 'localStorage', 'navigator'],
      caches: ['localStorage']
    }
  })

export default i18n
