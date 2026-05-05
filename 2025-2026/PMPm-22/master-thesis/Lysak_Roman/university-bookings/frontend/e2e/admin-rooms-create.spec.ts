import { test, expect, Page } from '@playwright/test';

const APP_URL = 'http://localhost:3000';
// Унікальна назва щоб не конфліктувати між запусками
const ROOM_NAME = `E2E Аудиторія ${Date.now()}`;

// ─────────────────────────────────────────────────────────────────────────────
// Хелпер: авторизація (повторює логіку auth-admin.spec.ts)
// ─────────────────────────────────────────────────────────────────────────────
async function ensureAdminLoggedIn(page: Page): Promise<void> {
  let userFromBackend: any = null;

  page.on('response', async response => {
    if (response.url().includes('/api/auth/login') && response.request().method() === 'POST') {
      try { userFromBackend = await response.json(); } catch {}
    }
  });

  await page.goto(APP_URL);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  await page.waitForTimeout(4_000);

  if (userFromBackend) {
    console.log('  ✅ Вже авторизовано як:', userFromBackend.email);
    return;
  }

  // Видаляємо протухлі MSAL-токени
  await page.evaluate(() => {
    Object.keys(localStorage).filter(k => k.startsWith('msal.')).forEach(k => localStorage.removeItem(k));
    sessionStorage.clear();
  });

  await page.goto(`${APP_URL}/login`);
  await page.waitForLoadState('networkidle', { timeout: 5_000 }).catch(() => {});

  const btn = page.getByRole('button', { name: /увійти|sign in|microsoft/i });
  await expect(btn).toBeVisible({ timeout: 5_000 });
  await btn.click();

  await page.waitForURL(/login\.microsoftonline\.com|microsoft\.com|live\.com/i, { timeout: 15_000 });

  console.log('\n  ════════════════════════════════════════════');
  console.log('  ⏳ Увійдіть через Microsoft і поверніться в додаток.');
  console.log('  Очікування до 2 хвилин...');
  console.log('  ════════════════════════════════════════════\n');

  await page.waitForURL(`${APP_URL}/**`, { timeout: 120_000 });
  await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => {});

  const deadline = Date.now() + 20_000;
  while (!userFromBackend && Date.now() < deadline) {
    await page.waitForTimeout(500);
  }

  if (!userFromBackend) {
    await page.reload();
    await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
    await page.waitForTimeout(3_000);
  }

  expect(userFromBackend, '/api/auth/login не отримано').not.toBeNull();
  expect(userFromBackend.role).toBe('Admin');
  console.log('  ✅ Авторизовано як Admin:', userFromBackend.email);
}

// ─────────────────────────────────────────────────────────────────────────────
// Хелпер: прокручує всі scrollable контейнери вниз
// ─────────────────────────────────────────────────────────────────────────────
async function scrollModalToBottom(page: Page): Promise<void> {
  await page.evaluate(() => {
    document.querySelectorAll<HTMLElement>('*').forEach(el => {
      const s = getComputedStyle(el).overflowY;
      if ((s === 'auto' || s === 'scroll') && el.scrollHeight > el.clientHeight) {
        el.scrollTop = el.scrollHeight;
      }
    });
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// ТЕСТ 1 — 24-годинний формат полів "Від" і "До"
// ─────────────────────────────────────────────────────────────────────────────
test('Тест 1 · Поля часу показують 24-годинний формат (без AM/PM)', async ({ page }) => {
  console.log('\n══════════════════════════════════════════════════');
  console.log('ТЕСТ 1 · 24-годинний формат часу');

  await ensureAdminLoggedIn(page);

  // Переходимо на сторінку кімнат
  await page.goto(`${APP_URL}/admin/rooms`);
  await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});
  console.log('  ✅ Відкрито /admin/rooms');

  // Відкриваємо модал
  await page.getByRole('button', { name: 'Додати приміщення' }).click();
  await page.waitForTimeout(500);

  // Прокручуємо до секції розкладу
  await scrollModalToBottom(page);
  await page.waitForTimeout(300);

  await page.screenshot({ path: 'e2e/screenshots/test1-time-inputs.png' });

  // Знаходимо поля "Від" і "До" (type="text", placeholder="ГГ:ХХ")
  const timeInputs = page.getByPlaceholder('ГГ:ХХ');
  const count = await timeInputs.count();
  console.log(`  Знайдено полів часу: ${count}`);
  expect(count).toBeGreaterThanOrEqual(2);

  const fromValue = await timeInputs.nth(0).inputValue();
  const toValue   = await timeInputs.nth(1).inputValue();
  console.log(`  Від = "${fromValue}", До = "${toValue}"`);

  // Значення НЕ повинні містити AM або PM
  expect(fromValue).not.toMatch(/AM|PM/i);
  expect(toValue).not.toMatch(/AM|PM/i);

  // Значення повинні бути у форматі HH:MM
  expect(fromValue).toMatch(/^\d{2}:\d{2}$/);
  expect(toValue).toMatch(/^\d{2}:\d{2}$/);

  console.log('  ✅ Обидва поля у 24-годинному форматі HH:MM');

  // Закриваємо діалог
  await page.getByRole('button', { name: 'Скасувати' }).click();

  console.log('══════════════════════════════════════════════════\n');
});

// ─────────────────────────────────────────────────────────────────────────────
// ТЕСТ 2 — Створення кімнати зі слотами розкладу
// ─────────────────────────────────────────────────────────────────────────────
test('Тест 2 · Створення кімнати з розкладом та перевірка в списку', async ({ page }) => {
  console.log('\n══════════════════════════════════════════════════');
  console.log('ТЕСТ 2 · Створення кімнати зі слотами розкладу');
  console.log(`  Назва: "${ROOM_NAME}"`);

  await ensureAdminLoggedIn(page);

  // ── Крок 1: Відкрити /admin/rooms ─────────────────────────────────────────
  await page.goto(`${APP_URL}/admin/rooms`);
  await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});

  // Запам'ятовуємо кількість кімнат до створення
  const countBefore = await page.locator('table tbody tr').count();
  console.log(`  Кімнат до створення: ${countBefore}`);

  await page.screenshot({ path: 'e2e/screenshots/test2-01-rooms-before.png' });

  // ── Крок 2: Відкрити модал ─────────────────────────────────────────────────
  await page.getByRole('button', { name: 'Додати приміщення' }).click();
  await page.waitForTimeout(500);
  await expect(page.getByRole('dialog')).toBeVisible({ timeout: 3_000 });
  console.log('  ✅ Модал відкрито');

  // ── Крок 3: Заповнити назву ────────────────────────────────────────────────
  await page.getByRole('textbox', { name: 'Назва' }).fill(ROOM_NAME);
  console.log(`  ✅ Введено назву: "${ROOM_NAME}"`);

  // ── Крок 4: Прокрутити до секції розкладу ─────────────────────────────────
  await scrollModalToBottom(page);
  await page.waitForTimeout(300);

  // Перевіряємо що секція розкладу видима
  await expect(page.getByText('Розклад доступності')).toBeVisible({ timeout: 3_000 });
  console.log('  ✅ Секція "Розклад доступності" видима');

  // ── Крок 5: Слот 1 — Пн 09:00–18:00 ──────────────────────────────────────
  const timeInputs = page.getByPlaceholder('ГГ:ХХ');

  // день "Пн" вже вибраний за замовчуванням (value=1)
  await timeInputs.nth(0).fill('09:00');
  await timeInputs.nth(1).fill('18:00');
  await page.getByRole('button', { name: '+ Додати' }).click();
  await page.waitForTimeout(300);

  // Перевіряємо що слот з'явився
  await expect(page.getByText('Пн · 09:00 – 18:00')).toBeVisible({ timeout: 3_000 });
  console.log('  ✅ Слот "Пн · 09:00 – 18:00" додано');

  await page.screenshot({ path: 'e2e/screenshots/test2-02-slot-mon.png' });

  // ── Крок 6: Слот 2 — Вт 10:00–17:00 ──────────────────────────────────────
  await page.getByLabel('День тижня').selectOption('Вт');
  await timeInputs.nth(0).fill('10:00');
  await timeInputs.nth(1).fill('17:00');
  await page.getByRole('button', { name: '+ Додати' }).click();
  await page.waitForTimeout(300);

  // Перевіряємо що обидва слоти видимі
  await expect(page.getByText('Пн · 09:00 – 18:00')).toBeVisible({ timeout: 3_000 });
  await expect(page.getByText('Вт · 10:00 – 17:00')).toBeVisible({ timeout: 3_000 });
  console.log('  ✅ Слот "Вт · 10:00 – 17:00" додано');

  await page.screenshot({ path: 'e2e/screenshots/test2-03-both-slots.png' });

  // ── Крок 7: Зберегти ──────────────────────────────────────────────────────
  // Слухаємо POST /api/admin/rooms
  const [createResponse] = await Promise.all([
    page.waitForResponse(
      r => r.url().includes('/api/admin/rooms') && r.request().method() === 'POST',
      { timeout: 10_000 }
    ),
    page.getByRole('button', { name: 'Зберегти' }).click(),
  ]);

  expect(createResponse.status()).toBeLessThan(400);
  console.log(`  ✅ POST /api/admin/rooms → ${createResponse.status()}`);

  // Чекаємо поки діалог закриється і дані перезавантажаться
  await expect(page.getByRole('dialog')).not.toBeVisible({ timeout: 5_000 });
  await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});

  await page.screenshot({ path: 'e2e/screenshots/test2-04-after-save.png' });

  // ── Крок 8: Знайти кімнату в таблиці ──────────────────────────────────────
  console.log('  Шукаємо кімнату в таблиці...');

  // Використовуємо пошук щоб знайти швидко (не переходити по сторінках)
  const searchBox = page.getByPlaceholder('Пошук за назвою або типом...');
  await searchBox.fill(ROOM_NAME);
  await page.waitForTimeout(400);

  const roomRow = page.getByRole('row', { name: new RegExp(ROOM_NAME) });
  await expect(roomRow).toBeVisible({ timeout: 5_000 });
  console.log('  ✅ Кімнату знайдено в таблиці');

  await page.screenshot({ path: 'e2e/screenshots/test2-05-room-in-table.png' });

  // ── Крок 9: Перевірити панель розкладу ────────────────────────────────────
  const scheduleBtn = roomRow.getByRole('button', { name: 'Розклад' });
  await scheduleBtn.click();
  await page.waitForTimeout(400);

  // Перевіряємо що панель з назвою кімнати відкрилась
  await expect(page.getByText(`Розклад доступності: ${ROOM_NAME}`)).toBeVisible({ timeout: 3_000 });

  // Перевіряємо обидва слоти
  await expect(page.getByText('Пн · 09:00 – 18:00')).toBeVisible({ timeout: 3_000 });
  await expect(page.getByText('Вт · 10:00 – 17:00')).toBeVisible({ timeout: 3_000 });

  console.log('  ✅ Обидва слоти відображаються в панелі розкладу');

  await page.screenshot({ path: 'e2e/screenshots/test2-06-schedule-panel.png' });

  // ── Крок 10 (cleanup): Видалити тестову кімнату ───────────────────────────
  console.log('  Видаляємо тестову кімнату...');

  // Закриваємо панель розкладу (натискаємо Розклад ще раз — toggle)
  await scheduleBtn.click();
  await page.waitForTimeout(300);

  const deleteBtn = roomRow.getByRole('button', { name: 'Вид.' });
  page.once('dialog', dialog => dialog.accept());
  await deleteBtn.click();
  await page.waitForTimeout(500);

  // Переконуємось що кімната зникла
  await expect(page.getByRole('row', { name: new RegExp(ROOM_NAME) })).not.toBeVisible({ timeout: 5_000 });
  console.log('  ✅ Тестова кімната видалена');

  await page.screenshot({ path: 'e2e/screenshots/test2-07-cleanup.png' });

  console.log('\n✅ Тест 2 пройшов повністю!');
  console.log('══════════════════════════════════════════════════\n');
});
