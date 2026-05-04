import { test, expect } from '@playwright/test';

const APP_URL = 'http://localhost:3000';

test('Admin login & role verification', async ({ page }) => {

  // ── STEP 1: Open app ───────────────────────────────────────────────────────
  console.log('\n══════════════════════════════════════════════════');
  console.log('STEP 1 · Opening app...');
  await page.goto(APP_URL);
  await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
  console.log('  Title:', await page.title());
  console.log('  URL:', page.url());

  // Start capturing /api/auth/login response from the very beginning
  let userFromBackend: any = null;
  page.on('response', async response => {
    if (response.url().includes('/api/auth/login') && response.request().method() === 'POST') {
      try {
        userFromBackend = await response.json();
        console.log('  [capture] /api/auth/login →', JSON.stringify(userFromBackend));
      } catch {}
    }
  });

  // ── STEP 2: Handle authentication ─────────────────────────────────────────
  console.log('\nSTEP 2 · Authentication...');

  // Wait a moment for React + MSAL to initialize and call /api/auth/login
  await page.waitForTimeout(4_000);

  if (userFromBackend) {
    console.log('  ✅ Already authenticated — session is active');
  } else {
    // Check if MSAL has tokens (may be expired)
    const hasMsal = await page.evaluate(() =>
      Object.keys(localStorage).some(k => k.startsWith('msal.'))
    );

    if (hasMsal) {
      console.log('  ⚠️  MSAL tokens found but session expired — clearing...');
      await page.evaluate(() => {
        Object.keys(localStorage).filter(k => k.startsWith('msal.')).forEach(k => localStorage.removeItem(k));
        sessionStorage.clear();
      });
    }

    // Navigate to login page
    await page.goto(`${APP_URL}/login`);
    await page.waitForLoadState('networkidle', { timeout: 5_000 }).catch(() => {});

    // Click the sign-in button
    const btn = page.getByRole('button', { name: /увійти|sign in|microsoft/i });
    await expect(btn).toBeVisible({ timeout: 5_000 });
    console.log('  Clicking "Увійти через Microsoft"...');
    await btn.click();

    // Wait for redirect to Microsoft login (URL leaves localhost)
    console.log('  Waiting for redirect to Microsoft login page...');
    await page.waitForURL(/login\.microsoftonline\.com|microsoft\.com|live\.com/i, { timeout: 15_000 });
    console.log('  ✅ On Microsoft login:', page.url().substring(0, 80) + '...');

    console.log('\n  ════════════════════════════════════════════');
    console.log('  ⏳ Please log in with:');
    console.log('     Email:    roman.lysak@lnu.edu.ua');
    console.log('     Password: (your password)');
    console.log('  Waiting up to 2 minutes...');
    console.log('  ════════════════════════════════════════════\n');

    // Wait for redirect BACK to our app (user completes login)
    await page.waitForURL(`${APP_URL}/**`, { timeout: 120_000 });
    console.log('  ✅ Redirected back to app:', page.url());

    // Wait for MSAL to finish processing + React to call /api/auth/login
    await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => {});

    // Additional wait for /api/auth/login to fire (up to 20 seconds)
    const deadline = Date.now() + 20_000;
    while (!userFromBackend && Date.now() < deadline) {
      await page.waitForTimeout(500);
    }

    if (!userFromBackend) {
      // If still not captured, try reloading — MSAL might need a page reload to settle
      console.log('  ⚠️  /api/auth/login not captured yet — reloading...');
      await page.reload();
      await page.waitForLoadState('networkidle', { timeout: 10_000 }).catch(() => {});
      await page.waitForTimeout(3_000);
    }
  }

  // ── STEP 3: Verify role ────────────────────────────────────────────────────
  console.log('\nSTEP 3 · Checking user role...');

  if (!userFromBackend) {
    await page.screenshot({ path: 'e2e/screenshots/debug-step3.png' });
    console.log('  ❌ /api/auth/login not captured. Debug screenshot saved.');
    console.log('     Current URL:', page.url());
  }

  expect(userFromBackend, '/api/auth/login was not captured').not.toBeNull();

  console.log('  email:        ', userFromBackend.email);
  console.log('  displayName:  ', userFromBackend.displayName);
  console.log('  role:         ', userFromBackend.role, `(${typeof userFromBackend.role})`);

  expect(
    userFromBackend.role,
    `Expected "Admin" but got "${userFromBackend.role}" (${typeof userFromBackend.role})`
  ).toBe('Admin');
  console.log('  ✅ Role = "Admin"');

  // ── STEP 4: Admin panel access ─────────────────────────────────────────────
  console.log('\nSTEP 4 · Admin panel pages:');

  const adminPages = [
    { path: '/admin',              label: 'Dashboard' },
    { path: '/admin/appointments', label: 'Appointments' },
    { path: '/admin/services',     label: 'Services' },
    { path: '/admin/staff',        label: 'Staff' },
    { path: '/admin/users',        label: 'Users' },
  ];

  const apiResults: { url: string; status: number }[] = [];
  page.on('response', r => {
    const url = r.url();
    if (url.includes('/api/admin') || url.includes('/api/appointments') || url.includes('/api/users') || url.includes('/api/services')) {
      apiResults.push({ url: url.replace('http://localhost:5000', ''), status: r.status() });
    }
  });

  for (const { path, label } of adminPages) {
    await page.goto(`${APP_URL}${path}`);
    await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});
    const accessible = page.url().includes(path);
    console.log(`  ${accessible ? '✅' : '❌'} ${label} → ${page.url()}`);
  }

  // ── STEP 5: Screenshot ─────────────────────────────────────────────────────
  console.log('\nSTEP 5 · Screenshot of admin dashboard...');
  await page.goto(`${APP_URL}/admin`);
  await page.waitForLoadState('networkidle', { timeout: 8_000 }).catch(() => {});
  await page.screenshot({ path: 'e2e/screenshots/admin-dashboard.png', fullPage: true });
  console.log('  📸 Saved: e2e/screenshots/admin-dashboard.png');

  // ── STEP 6: API summary ────────────────────────────────────────────────────
  if (apiResults.length > 0) {
    console.log('\nSTEP 6 · API calls observed:');
    for (const r of apiResults) {
      console.log(`  ${r.status < 400 ? '✅' : '❌'} [${r.status}] ${r.url}`);
    }
    const failed = apiResults.filter(r => r.status >= 400);
    if (failed.length > 0) {
      console.log(`  ⚠️  ${failed.length} failed API call(s)`);
    }
  }

  console.log('\n══════════════════════════════════════════════════');
  await expect(page).toHaveURL(/\/admin/);
  console.log('✅ All checks passed!');
  console.log('══════════════════════════════════════════════════\n');
});
