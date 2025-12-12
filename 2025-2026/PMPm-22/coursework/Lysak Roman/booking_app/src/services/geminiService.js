import { GoogleGenAI } from '@google/genai';
import Fuse from 'fuse.js'

const API_KEY = process.env.REACT_APP_GEMINI_API_KEY;
let genAI;

if (API_KEY) {
  genAI = new GoogleGenAI({ apiKey: API_KEY });
}

const findServiceInBookings = (serviceName, bookingsData) => {
  
  const allServices = [];
  for (const booking of bookingsData) {
    allServices.push({
      booking,
      name: booking.displayName
    });
  }

  const options = {
    keys: ['name'],
    threshold: 0.4, 
    minMatchCharLength: 2,
  };

  const fuse = new Fuse(allServices, options);

  const result = fuse.search(serviceName);

  if (result.length > 0) {
    const { booking } = result[0].item;
    return { booking };
  }
  return null;
};

const formatDuration = (duration) => {
  if (!duration || duration === 'Не вказано') return duration;
  const match = duration.match(/PT(?:(\d+)H)?(?:(\d+)M)?/);
  if (!match) return duration;
  const hours = match[1] ? `${match[1]} год` : '';
  const minutes = match[2] ? `${match[2]} хв` : '';
  return `${hours} ${minutes}`.trim() || duration;
};

export const sendMessageToGemini = async (userMessage, bookingsData, userAppointments = [], onCancelAppointment = null) => {
  if (!genAI) {
    throw new Error('Gemini API key not configured. Please add REACT_APP_GEMINI_API_KEY to .env.local');
  }

  console.log('Bookings data received:', bookingsData);
  console.log('User appointments received:', userAppointments);

  if (!bookingsData || bookingsData.length === 0) {
    return 'Вибачте, зараз немає доступних послуг для резервування. Будь ласка, спробуйте пізніше.';
  }

  const nextAppointmentMatch = userMessage.match(/(?:коли|яка)\s+(?:моя\s+)?(?:найближча|наступна)\s+(?:зустріч|appointment)/i);
  if (nextAppointmentMatch || userMessage.match(/(?:у\s+мене\s+)?(?:щось\s+)?(?:сьогодні|завтра)/i)) {
    if (!userAppointments || userAppointments.length === 0) {
      return '📅 У вас наразі немає запланованих зустрічей. Бажаєте зарезервувати послугу?';
    }

    const now = new Date();
    const futureAppointments = userAppointments.filter(appt => {
      const apptDate = new Date(appt.startDateTime?.dateTime);
      return apptDate > now;
    });

    if (futureAppointments.length === 0) {
      return '📅 У вас немає майбутніх зустрічей. Всі ваші зустрічі вже минули. Бажаєте зарезервувати нову послугу?';
    }

    const nextAppt = futureAppointments[0]; 
    const startDate = new Date(nextAppt.startDateTime?.dateTime);
    const endDate = new Date(nextAppt.endDateTime?.dateTime);

    const formattedDate = startDate.toLocaleDateString('uk-UA', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
      weekday: 'long',
    });
    const formattedTime = `${startDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' })} - ${endDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' })}`;

    const timeUntil = startDate - now;
    const hoursUntil = Math.floor(timeUntil / (1000 * 60 * 60));
    const daysUntil = Math.floor(hoursUntil / 24);

    let timeUntilText = '';
    if (daysUntil > 0) {
      timeUntilText = `За ${daysUntil} ${daysUntil === 1 ? 'день' : daysUntil < 5 ? 'дні' : 'днів'}`;
    } else if (hoursUntil > 0) {
      const remainingHours = hoursUntil % 24;
      timeUntilText = `За ${remainingHours} ${remainingHours === 1 ? 'годину' : remainingHours < 5 ? 'години' : 'годин'}`;
    } else {
      const minutesUntil = Math.floor(timeUntil / (1000 * 60));
      timeUntilText = `За ${minutesUntil} хвилин`;
    }

    return `🎯 **Ваша найближча зустріч:**

**${nextAppt.serviceName}** в ${nextAppt.bookingName}
📅 ${formattedDate}
⏰ ${formattedTime}
⌛ ${timeUntilText}
${nextAppt.customerPhone ? `📱 ${nextAppt.customerPhone}` : ''}

✨ Бажаєте переглянути всі зустрічі або щось змінити?`;
  }

  const cancelMatch = userMessage.match(/(?:скасувати|видалити|cancel)\s+(?:зустріч|appointment|бронювання)(?:\s+(\d+))?/i);
  if (cancelMatch || userMessage.match(/(?:хочу\s+)?(?:скасувати|відмінити)/i)) {
    if (!userAppointments || userAppointments.length === 0) {
      return '📅 У вас немає зустрічей для скасування.';
    }

    const appointmentNumber = cancelMatch ? parseInt(cancelMatch[1]) : null;

    if (appointmentNumber && appointmentNumber > 0 && appointmentNumber <= userAppointments.length) {
      const apptToCancel = userAppointments[appointmentNumber - 1];
      
      if (onCancelAppointment) {
        try {
          await onCancelAppointment(apptToCancel.id, apptToCancel.bookingId);
          return `✅ Зустріч **"${apptToCancel.serviceName}"** успішно скасовано!\n\nБажаєте переглянути інші зустрічі або зарезервувати нову послугу?`;
        } catch (error) {
          return `❌ Не вдалося скасувати зустріч: ${error.message}`;
        }
      } else {
        return '❌ Функція скасування недоступна. Будь ласка, спробуйте пізніше.';
      }
    } else {
      const now = new Date();
      const futureAppointments = userAppointments.filter(appt => {
        const apptDate = new Date(appt.startDateTime?.dateTime);
        return apptDate > now;
      });

      if (futureAppointments.length === 0) {
        return '📅 У вас немає майбутніх зустрічей для скасування. Всі ваші зустрічі вже минули.';
      }

      const appointmentsList = futureAppointments.map((appt, index) => {
        const startDate = new Date(appt.startDateTime?.dateTime);
        const formattedDate = startDate.toLocaleDateString('uk-UA', {
          day: 'numeric',
          month: 'long',
          year: 'numeric',
        });
        const formattedTime = startDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' });

        return `${index + 1}. **${appt.serviceName}** в ${appt.bookingName}\n   📅 ${formattedDate} о ${formattedTime}`;
      }).join('\n\n');

      return `🗑️ **Які зустрічі ви хочете скасувати?**\n\n${appointmentsList}\n\n💡 Щоб скасувати зустріч, напишіть: "скасувати зустріч [номер]"\nНаприклад: "скасувати зустріч 1"`;
    }
  }

  const appointmentMatch = userMessage.match(/які\s+(зустрічі|appointments|бронювання)\s+(я\s+маю|в\s+мене|у\s+мене)/i);
  if (appointmentMatch || userMessage.match(/мої\s+(зустрічі|appointments|бронювання)/i)) {
    if (!userAppointments || userAppointments.length === 0) {
      return '📅 У вас наразі немає запланованих зустрічей. Бажаєте зарезервувати послугу?';
    }

    const appointmentsInfo = userAppointments.map((appt, index) => {
      const startDate = new Date(appt.startDateTime?.dateTime);
      const endDate = new Date(appt.endDateTime?.dateTime);
      const formattedDate = startDate.toLocaleDateString('uk-UA', {
        day: 'numeric',
        month: 'long',
        year: 'numeric',
      });
      const formattedTime = `${startDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' })} - ${endDate.toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' })}`;

      return `${index + 1}. **${appt.serviceName}** в ${appt.bookingName}
   📅 Дата: ${formattedDate}
   ⏰ Час: ${formattedTime}
   📧 Email: ${appt.customerEmailAddress}
   ${appt.customerPhone ? `📱 Телефон: ${appt.customerPhone}` : ''}`;
    }).join('\n\n');

    const prompt = `
Ти - асистент з бронювання. Користувач запитує про свої зустрічі.
Ось список зустрічей користувача:

${appointmentsInfo}

Дай дружню відповідь українською з емодзі, підтвердивши що це всі його зустрічі.
Запропонуй допомогу якщо користувач хоче щось змінити або додати нову зустріч.
    `;

    try {
      const result = await genAI.models.generateContent({
        model: 'gemini-2.5-flash-lite',
        contents: prompt,
      });
      return result.text || result;
    } catch (error) {
      console.error('Error calling Gemini API:', error);
      return `📅 **Ваші заплановані зустрічі:**\n\n${appointmentsInfo}\n\n✨ Бажаєте додати нову зустріч або змінити існуючу?`;
    }
  }

  const serviceSearchMatch =
    userMessage.match(/де\s+(я\s+)?можу\s+знайти\s+(.+)/i) ||
    userMessage.match(/де\s+(знайти|шукати)\s+(.+)/i) ||
    userMessage.match(/як\s+знайти\s+(.+)/i) ||
    userMessage.match(/хочу\s+знайти\s+(.+)/i) ||
    userMessage.match(/де\s+є\s+(.+)/i) ||
    userMessage.match(/де\s+доступна\s+(.+)/i) ||
    userMessage.match(/де\s+можна\s+записатися\s+на\s+(.+)/i) ||
    userMessage.match(/де\s+(.+)/i) ||
    userMessage.match(/де\s+запис\s+на\s+(.+)/i);
    
  if (serviceSearchMatch) {
    const serviceName = serviceSearchMatch[1].trim();
    const found = findServiceInBookings(serviceName, bookingsData);

    if (found) {
      const { booking } = found;
      const bookingUrl = booking.webSiteUrl || `https://outlook.office.com/book/${booking.id}/`;
      const prompt = `
Ти - асистент з бронювання. Користувач питає, де знайти послугу "${booking.displayName}".
Ось інформація про цю послугу:
Бізнес: ${booking.displayName}
Посилання для бронювання: ${bookingUrl}

Дай відповідь українською, з емодзі, з короткою інструкцією як забронювати цю послугу.
      `;
      try {
        console.log('Sending prompt to Gemini (service search)...');
        const result = await genAI.models.generateContent({
          model: 'gemini-2.5-flash-lite',
          contents: prompt,
        });
        console.log('Gemini response received:', result);
        return result.text || result;
      } catch (error) {
        console.error('Error calling Gemini API:', error);
        if (error.message?.includes('API key')) {
          throw new Error('Помилка автентифікації API. Перевірте ваш API ключ.');
        } else if (error.message?.includes('quota')) {
          throw new Error('Перевищено ліміт запитів. Спробуйте через хвилину.');
        } else {
          throw new Error('Вибачте, виникла помилка при обробці вашого запиту. Спробуйте ще раз.');
        }
      }
    } else {
      return 'Вибачте, такої послуги не знайдено. Ось список доступних послуг. Якщо потрібно, уточніть назву послуги.';
    }
  }
  const servicesContext = bookingsData.map((booking) => {
    const bookingUrl = booking.publicUrl || `https://outlook.office.com/book/${booking.id}/`;

    return {
      businessId: booking.id,
      businessName: booking.displayName,
      bookingUrl: bookingUrl,
      description: booking.description || '',
      phone: booking.phone || '',
      email: booking.email || '',
      address: booking.address ? `${booking.address.street}, ${booking.address.city}` : '',
      services: booking.services?.map((service) => ({
        name: service.displayName,
        description: service.description || '',
        duration: service.defaultDuration || 'Не вказано',
        price: service.defaultPrice !== undefined ? `${service.defaultPrice} грн` : 'Безкоштовно',
      })) || [],
    };
  });

  const formattedServices = servicesContext.map((booking) => {
    const servicesList = booking.services.map((service) =>
      `   - ${service.name}: ${service.description || 'Опис відсутній'}. Тривалість: ${formatDuration(service.duration)}, Ціна: ${service.price}`
    ).join('\n');

    return `Бізнес: ${booking.businessName}
Опис: ${booking.description}
Контакти: ${booking.phone || ''} ${booking.email || ''}
Адреса: ${booking.address}
Посилання для бронювання: ${booking.bookingUrl}
Доступні послуги:
${servicesList}`;
  }).join('\n\n---\n\n');

  const systemPrompt = `Ти - дружній асистент з резервування для системи бронювання. Твоє завдання - допомагати користувачам знайти та зарезервувати послуги.

ДОСТУПНІ БІЗНЕСИ ТА ПОСЛУГИ:

${formattedServices}

ВАЖЛИВІ ПРАВИЛА:
1. ЗАВЖДИ відповідай українською мовою
2. Будь ввічливим, дружнім та допомоглим
3. Використовуй емодзі для покращення комунікації (наприклад: ✅, 📅, 💰, 🏢, 📍)

ВІДПОВІДІ НА ТИПОВІ ЗАПИТАННЯ:

Якщо користувач питає "що я можу зарезервувати" або "які послуги доступні":
- Покажи список ВСІХ доступних послуг групуючи їх по бізнесам
- Для КОЖНОЇ послуги вкажи: назву, короткий опис, тривалість та ціну
- В кінці додай посилання для бронювання

Якщо користувач питає "Де я можу знайти [назва послуги]":
- Знайди цю послугу в списку доступних послуг
- Вкажи назву бізнесу, що надає цю послугу
- Дай посилання для бронювання у форматі: [Перейти до бронювання](URL)
- Додай короткі інструкції як зробити резервацію

Якщо користувач питає "Як зробити резервацію":
- Поясни покрокову інструкцію:
  1. Перейти за посиланням для бронювання
  2. Обрати послугу
  3. Вибрати дату та час
  4. Заповнити контактні дані
  5. Підтвердити бронювання

Якщо питання не стосується бронювання:
- Ввічливо поясни, що ти можеш допомогти лише з резервуванням послуг
- Запропонуй показати доступні послуги

ФОРМАТ ВІДПОВІДЕЙ:
- Використовуй маркований список для переліку послуг
- Для посилань використовуй формат: [Текст посилання](URL)
- Структуруй відповідь чітко та читабельно`;

  const fullPrompt = `${systemPrompt}

Запитання користувача: "${userMessage}"

Твоя відповідь (українською, з емодзі):`;

  try {
    console.log('Sending prompt to Gemini...');
    const result = await genAI.models.generateContent({
      model: 'gemini-2.5-flash-lite',
      contents: fullPrompt,
    });
    console.log('Gemini response received:', result);
    return result.text || result;
  } catch (error) {
    console.error('Error calling Gemini API:', error);
    console.error('Error details:', error.message);
    console.error('Error response:', error.response);

    if (error.message?.includes('API key')) {
      throw new Error('Помилка автентифікації API. Перевірте ваш API ключ.');
    } else if (error.message?.includes('quota')) {
      throw new Error('Перевищено ліміт запитів. Спробуйте через хвилину.');
    } else {
      throw new Error('Вибачте, виникла помилка при обробці вашого запиту. Спробуйте ще раз.');
    }
  }
};

export const isApiKeyConfigured = () => {
  return API_KEY && API_KEY !== 'your_gemini_api_key_here';
};
