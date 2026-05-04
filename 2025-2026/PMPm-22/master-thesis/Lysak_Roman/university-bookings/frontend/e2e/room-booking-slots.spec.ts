import { test, expect, Page, Response } from '@playwright/test';

const APP_URL = 'http://localhost:3000';
const ROOM_ID = 'c4b45e19-be45-4c80-af63-0f9ed65e19de';
const ROOM_URL = `${APP_URL}/rooms/${ROOM_ID}`;

// Availability window for test3: 08:00 – 20:00 (720 min) Mon–Fri
// Expected total slot counts per duration (with zero existing bookings):
//   40 min → 720/40 = 18  (08:00 … 19:20)
//   60 min → 720/60 = 12  (08:00 … 19:00)
//   80 min → 720/80 = 9   (08:00 … 18:40)
//  120 min → 720/120 = 6  (08:00 … 18:00)

type Slot = {
  startTime: string;  // ISO 8601 with offset
  endTime: string;
  available: boolean;
  isHeld: boolean;
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

function isSlotsUrl(url: string): boolean {
  return url.includes(`/api/rooms/${ROOM_ID}/available-slots`);
}

/**
 * Navigate to the room detail page and wait for all initial API calls to settle.
 *
 * On load, the component fires TWO available-slots calls:
 *   1. For today (may be weekend → empty response)
 *   2. For the first available weekday (after room data loads and selectedDate updates)
 * waitForLoadState('networkidle') ensures both calls complete before the test runs.
 */
async function openRoomPage(page: Page): Promise<void> {
  await page.goto(ROOM_URL);
  await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => {});
}

/**
 * Locator for the date-picker buttons (пн, вт, ср, чт, пт, сб, нд).
 * Uses text-content filtering because makeStyles generates hashed class names.
 */
function datePicker(page: Page) {
  return page.getByRole('button').filter({ hasText: /^(пн|вт|ср|чт|пт|сб|нд)/ });
}

/**
 * Change the duration selector and wait for the new slots API response.
 * Filters by `duration=<value>` in the query string so it doesn't match a
 * stale request from a previous selection.
 */
async function selectDuration(page: Page, durationValue: string): Promise<Slot[]> {
  const [response] = await Promise.all([
    page.waitForResponse(
      (r: Response) =>
        isSlotsUrl(r.url()) && r.url().includes(`duration=${durationValue}`),
      { timeout: 10_000 },
    ),
    page.locator('select').selectOption(durationValue),
  ]);
  return (response as Response).json() as Promise<Slot[]>;
}

/**
 * Verify that consecutive slots in the API response are exactly `stepMin` apart.
 * Uses raw ISO timestamps, so it is timezone-safe.
 */
function assertSlotStep(slots: Slot[], stepMin: number): void {
  expect(slots.length).toBeGreaterThan(1);
  for (let i = 1; i < slots.length; i++) {
    const prev = new Date(slots[i - 1].startTime).getTime();
    const curr = new Date(slots[i].startTime).getTime();
    const actualMin = (curr - prev) / 60_000;
    expect(
      actualMin,
      `Крок між слотами ${i - 1}→${i}: очікувалось ${stepMin} хв, отримано ${actualMin} хв`,
    ).toBe(stepMin);
  }
}

/**
 * Verify that slot[0].startTime and slot[last].startTime display as
 * `expectedFirst` / `expectedLast` in the browser's local timezone
 * (uk-UA locale, HH:MM format).
 *
 * Note: assumes the test browser runs in the Kyiv timezone (UTC+3 / EET+DST),
 * which matches the server's "FLE Standard Time" offset used when building slots.
 */
function assertFirstLast(slots: Slot[], expectedFirst: string, expectedLast: string): void {
  const fmt = (iso: string) =>
    new Date(iso).toLocaleTimeString('uk-UA', { hour: '2-digit', minute: '2-digit' });

  expect(fmt(slots[0].startTime)).toBe(expectedFirst);
  expect(fmt(slots[slots.length - 1].startTime)).toBe(expectedLast);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

test.describe('RoomDetail · test3 · слоти бронювання', () => {

  // ── Тест 1: Базова структура сторінки ──────────────────────────────────────
  test('Тест 1 · Структура сторінки — назва, відповідальна особа, 4 варіанти тривалості', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 1 · Базова структура сторінки');

    await openRoomPage(page);

    // Назва приміщення
    await expect(page.getByText('test3', { exact: true })).toBeVisible({ timeout: 5_000 });
    console.log('  ✅ Назва "test3" відображається');

    // Відповідальна особа
    await expect(page.getByText('Лисак Роман')).toBeVisible({ timeout: 5_000 });
    console.log('  ✅ "Лисак Роман" відображається');

    // Вибір тривалості — 4 опції з правильними значеннями
    const select = page.locator('select');
    await expect(select).toBeVisible();
    const options = select.locator('option');
    await expect(options).toHaveCount(4);

    const values = await options.evaluateAll(
      (opts: HTMLOptionElement[]) => opts.map((o) => o.value),
    );
    expect(values).toEqual(['40', '60', '80', '120']);
    console.log('  ✅ 4 варіанти тривалості: 40, 60, 80, 120 хв');

    // За замовчуванням обрано 60 хв
    await expect(select).toHaveValue('60');
    console.log('  ✅ За замовчуванням: 60 хв');

    // Секція вибору дати присутня
    await expect(page.getByText('Оберіть дату')).toBeVisible();
    await expect(page.getByText('Оберіть час')).toBeVisible();
    console.log('  ✅ Секції "Оберіть дату" та "Оберіть час" відображаються');

    await page.screenshot({ path: 'e2e/screenshots/booking-test1-page-structure.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 2: 40 хв → 18 слотів з кроком 40 хв ─────────────────────────────
  test('Тест 2 · Тривалість 40 хв → 18 слотів, крок 40 хв, 08:00–19:20', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 2 · Слоти 40 хв');

    await openRoomPage(page);
    // Default is 60 min; switch to 40 → triggers new API call for correct weekday
    const slots = await selectDuration(page, '40');
    console.log(`  API: ${slots.length} слотів`);

    expect(slots).toHaveLength(18);
    assertSlotStep(slots, 40);
    assertFirstLast(slots, '08:00', '19:20');
    console.log('  ✅ 18 слотів, крок 40 хв, 08:00 → 19:20');

    // Перший і останній слот відображаються на сторінці
    await expect(
      page.getByRole('button', { name: '08:00' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    await expect(
      page.getByRole('button', { name: '19:20' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    console.log('  ✅ Кнопки 08:00 та 19:20 відображаються');

    // Кнопки 09:00 / 10:00 (рівні для 60 хв) НЕ є єдиними — між ними 08:40 / 09:20
    await expect(
      page.getByRole('button', { name: '08:40' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    await expect(
      page.getByRole('button', { name: '09:20' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    console.log('  ✅ Проміжні слоти 08:40 та 09:20 відображаються');

    await page.screenshot({ path: 'e2e/screenshots/booking-test2-40min-slots.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 3: 60 хв → 12 слотів з кроком 60 хв ─────────────────────────────
  test('Тест 3 · Тривалість 60 хв → 12 слотів, крок 60 хв, 08:00–19:00', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 3 · Слоти 60 хв');

    await openRoomPage(page);
    // Page loads with 60 min by default, but we can't capture that initial response
    // because the component first fires for today (may be weekend → empty), then
    // re-fires for the first available weekday.  Switch 40→60 to get a clean capture.
    await selectDuration(page, '40');
    const slots = await selectDuration(page, '60');
    console.log(`  API: ${slots.length} слотів`);

    expect(slots).toHaveLength(12);
    assertSlotStep(slots, 60);
    assertFirstLast(slots, '08:00', '19:00');
    console.log('  ✅ 12 слотів, крок 60 хв, 08:00 → 19:00');

    // Перевірка що щогодинні слоти відображаються
    for (const time of ['08:00', '09:00', '10:00', '18:00', '19:00']) {
      await expect(
        page.getByRole('button', { name: time }).first(),
      ).toBeVisible({ timeout: 5_000 });
    }
    console.log('  ✅ Погодинні кнопки 08:00–19:00 відображаються');

    // Слот 08:40 (характерний для 40 хв) не повинен бути присутній
    await expect(
      page.getByRole('button', { name: '08:40' }),
    ).toHaveCount(0);
    console.log('  ✅ Слоту 08:40 немає (не характерний для 60 хв)');

    await page.screenshot({ path: 'e2e/screenshots/booking-test3-60min-slots.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 4: 80 хв → 9 слотів з кроком 80 хв ──────────────────────────────
  test('Тест 4 · Тривалість 80 хв → 9 слотів, крок 80 хв, 08:00–18:40', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 4 · Слоти 80 хв');

    await openRoomPage(page);
    const slots = await selectDuration(page, '80');
    console.log(`  API: ${slots.length} слотів`);

    expect(slots).toHaveLength(9);
    assertSlotStep(slots, 80);
    assertFirstLast(slots, '08:00', '18:40');
    console.log('  ✅ 9 слотів, крок 80 хв, 08:00 → 18:40');

    // Перший і останній слот відображаються
    await expect(
      page.getByRole('button', { name: '08:00' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    await expect(
      page.getByRole('button', { name: '18:40' }).first(),
    ).toBeVisible({ timeout: 5_000 });

    // 09:20 є (другий слот при 80 хв), а 09:00 — ні
    await expect(
      page.getByRole('button', { name: '09:20' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    await expect(
      page.getByRole('button', { name: '09:00' }),
    ).toHaveCount(0);
    console.log('  ✅ Слот 09:20 є, 09:00 відсутній (правильний крок 80 хв)');

    await page.screenshot({ path: 'e2e/screenshots/booking-test4-80min-slots.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 5: 120 хв → 6 слотів з кроком 120 хв ────────────────────────────
  test('Тест 5 · Тривалість 120 хв → 6 слотів, крок 120 хв, 08:00–18:00', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 5 · Слоти 120 хв');

    await openRoomPage(page);
    const slots = await selectDuration(page, '120');
    console.log(`  API: ${slots.length} слотів`);

    expect(slots).toHaveLength(6);
    assertSlotStep(slots, 120);
    assertFirstLast(slots, '08:00', '18:00');
    console.log('  ✅ 6 слотів, крок 120 хв, 08:00 → 18:00');

    for (const time of ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00']) {
      await expect(
        page.getByRole('button', { name: time }).first(),
      ).toBeVisible({ timeout: 5_000 });
    }
    console.log('  ✅ Всі 6 парних кнопок відображаються: 08:00 10:00 12:00 14:00 16:00 18:00');

    // 19:00 не повинен бути (занадто пізно для 120 хв)
    await expect(
      page.getByRole('button', { name: '19:00' }),
    ).toHaveCount(0);
    console.log('  ✅ Слоту 19:00 немає (недостатньо часу для 2 год)');

    await page.screenshot({ path: 'e2e/screenshots/booking-test5-120min-slots.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 6: Вибір слоту відображає форму бронювання ──────────────────────
  test('Тест 6 · Вибір слоту показує форму; незалогінений — кнопка "Увійти"', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 6 · Форма бронювання');

    await openRoomPage(page);

    // До вибору слоту — textarea та кнопка бронювання відсутні
    await expect(page.locator('textarea')).not.toBeVisible();
    await expect(page.getByRole('button', { name: /бронювання|увійти для/i })).not.toBeVisible();
    console.log('  ✅ До вибору — форма прихована');

    // Вибираємо перший слот (08:00 для 60 хв)
    await page.getByRole('button', { name: '08:00' }).first().click();
    await page.waitForTimeout(500);

    // Після вибору — textarea з'являється
    await expect(page.locator('textarea')).toBeVisible({ timeout: 3_000 });
    console.log('  ✅ Після вибору — textarea відображається');

    // Кнопка дії — "Увійти для бронювання" (не залогінений)
    const actionBtn = page.getByRole('button', { name: /увійти для бронювання|підтвердити бронювання/i });
    await expect(actionBtn).toBeVisible({ timeout: 3_000 });
    console.log(`  ✅ Кнопка дії відображається: "${await actionBtn.innerText()}"`);

    // Кнопка 08:00 в активному стані (primary appearance)
    const activeSlot = page.getByRole('button', { name: '08:00' }).first();
    const isPrimary = await activeSlot.evaluate(
      (el: HTMLButtonElement) => el.getAttribute('appearance') === 'primary' || el.classList.contains('fui-Button--primary'),
    );
    console.log(`  Слот 08:00 активний (primary): ${isPrimary}`);

    await page.screenshot({ path: 'e2e/screenshots/booking-test6-slot-selected.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 7: Зміна тривалості скидає вибраний слот ────────────────────────
  test('Тест 7 · Зміна тривалості скидає вибраний слот і ховає форму', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 7 · Скидання вибору при зміні тривалості');

    await openRoomPage(page);

    // Вибираємо слот 08:00
    await page.getByRole('button', { name: '08:00' }).first().click();
    await page.waitForTimeout(500);
    await expect(page.locator('textarea')).toBeVisible({ timeout: 3_000 });
    console.log('  ✅ Слот вибрано, форма відображається');

    // Змінюємо тривалість на 40 хв
    await selectDuration(page, '40');
    await page.waitForTimeout(500);

    // Форма повинна зникнути — вибір скинуто
    await expect(page.locator('textarea')).not.toBeVisible({ timeout: 3_000 });
    await expect(page.getByRole('button', { name: /бронювання|увійти для/i })).not.toBeVisible();
    console.log('  ✅ Після зміни тривалості — форма прихована, вибір скинуто');

    // Слоти оновились до 40 хв — з'явились проміжні часи
    await expect(
      page.getByRole('button', { name: '08:40' }).first(),
    ).toBeVisible({ timeout: 5_000 });
    console.log('  ✅ Слоти оновились до 40 хв (08:40 відображається)');

    await page.screenshot({ path: 'e2e/screenshots/booking-test7-duration-clears-slot.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 8: Зміна дати скидає вибраний слот ──────────────────────────────
  test('Тест 8 · Зміна дати скидає вибраний слот', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 8 · Скидання вибору при зміні дати');

    await openRoomPage(page);

    // Вибираємо слот
    await page.getByRole('button', { name: '08:00' }).first().click();
    await page.waitForTimeout(500);
    await expect(page.locator('textarea')).toBeVisible({ timeout: 3_000 });
    console.log('  ✅ Слот вибрано');

    // Знаходимо другу доступну кнопку дати і клікаємо на неї
    // datePicker() uses text-content filtering (makeStyles hashes class names)
    const availableDateButtons = datePicker(page).and(page.locator('button:not([disabled])'));
    const count = await availableDateButtons.count();
    expect(count).toBeGreaterThan(1);

    const [dateChangeResponse] = await Promise.all([
      page.waitForResponse((r: Response) => isSlotsUrl(r.url()), { timeout: 10_000 }),
      availableDateButtons.nth(1).click(),
    ]);
    expect(dateChangeResponse.status()).toBeLessThan(400);
    await page.waitForTimeout(500);

    // Форма повинна зникнути
    await expect(page.locator('textarea')).not.toBeVisible({ timeout: 3_000 });
    console.log('  ✅ Після зміни дати — форма прихована, вибір скинуто');

    await page.screenshot({ path: 'e2e/screenshots/booking-test8-date-clears-slot.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 9: Вихідні дні (сб/нд) вимкнені ────────────────────────────────
  test('Тест 9 · Кнопки вихідних (сб/нд) вимкнені, доступні дні пн–пт', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 9 · Вихідні дні вимкнені');

    await openRoomPage(page);

    // datePicker() targets buttons by day-name text (makeStyles hashes class names)
    const allDateButtons = datePicker(page);
    const total = await allDateButtons.count();
    console.log(`  Всього кнопок дат: ${total}`);

    let weekendDisabled = 0;
    let weekdayEnabled = 0;

    for (let i = 0; i < total; i++) {
      const btn = allDateButtons.nth(i);
      const label = (await btn.innerText()).replace('\n', ' ');
      const disabled = await btn.isDisabled();

      if (label.startsWith('нд') || label.startsWith('сб')) {
        expect(disabled, `Вихідний день "${label}" має бути вимкнений`).toBe(true);
        weekendDisabled++;
        console.log(`  ✅ Вимкнено: "${label}"`);
      } else {
        // пн, вт, ср, чт, пт — мають бути активні
        expect(disabled, `Будній день "${label}" не повинен бути вимкнений`).toBe(false);
        weekdayEnabled++;
      }
    }

    expect(weekendDisabled).toBeGreaterThan(0);
    expect(weekdayEnabled).toBeGreaterThan(0);
    console.log(`  ✅ Вимкнено вихідних: ${weekendDisabled}, активних будніх: ${weekdayEnabled}`);

    await page.screenshot({ path: 'e2e/screenshots/booking-test9-weekends-disabled.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

  // ── Тест 10: API endpoint повертає правильні параметри ────────────────────
  test('Тест 10 · API available-slots передає date і duration правильно', async ({ page }) => {
    console.log('\n══════════════════════════════════════════════════');
    console.log('ТЕСТ 10 · Параметри API запиту');

    await openRoomPage(page);

    const capturedUrls: string[] = [];
    page.on('response', (r: Response) => {
      if (isSlotsUrl(r.url())) capturedUrls.push(r.url());
    });

    // Змінюємо тривалість
    await selectDuration(page, '120');

    const lastUrl = capturedUrls[capturedUrls.length - 1] ?? '';
    console.log(`  Останній API URL: ${lastUrl}`);

    expect(lastUrl).toContain(`/api/rooms/${ROOM_ID}/available-slots`);
    expect(lastUrl).toContain('duration=120');
    expect(lastUrl).toMatch(/date=\d{4}-\d{2}-\d{2}/);
    console.log('  ✅ URL містить roomId, duration=120, date=YYYY-MM-DD');

    await page.screenshot({ path: 'e2e/screenshots/booking-test10-api-params.png' });
    console.log('══════════════════════════════════════════════════\n');
  });

});
