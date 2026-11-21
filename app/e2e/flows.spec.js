import { expect, test } from '@playwright/test';
import { resolve } from 'node:path';
import { ADMIN } from '../playwright.config.js';

const PHOTO = resolve(import.meta.dirname, '../../tests/fixtures/pothole.jpg');
let seq = 0;
const uniqueEmail = () => `citizen${Date.now()}${seq++}@e2e.test`;

async function newCitizen(browser, { geo } = {}) {
  const context = await browser.newContext(geo
    ? { geolocation: { latitude: geo[0], longitude: geo[1], accuracy: geo[2] ?? 8 }, permissions: ['geolocation'] }
    : {});
  if (!geo) {
    // Simulate a user who denied location access.
    await context.addInitScript(() => {
      navigator.geolocation.getCurrentPosition = (_ok, fail) => setTimeout(() => fail({ code: 1, message: 'denied' }), 10);
    });
  }
  const page = await context.newPage();
  await page.goto('/register');
  await page.getByLabel('Name').fill('E2E Citizen');
  await page.getByLabel('Email').fill(uniqueEmail());
  await page.getByLabel('Password').fill('citizen-pass-1');
  await page.getByRole('button', { name: 'Create account' }).click();
  await expect(page.getByRole('heading', { name: 'Spotted a pothole?' })).toBeVisible();
  return { context, page };
}

async function startReport(page) {
  await page.goto('/report');
  await page.getByTestId('gallery-input').setInputFiles(PHOTO);
}

async function finishReport(page, { type } = {}) {
  await expect(page.getByRole('button', { name: 'Submit report' })).toBeVisible();
  if (type) await page.getByRole('button', { name: type }).click();
  await page.getByRole('button', { name: 'Submit report' }).click();
  const link = page.getByRole('link', { name: 'View hazard page' });
  await expect(link).toBeVisible();
  return Number((await link.getAttribute('href')).split('/').pop());
}

async function adminPage(browser) {
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto('/login');
  await page.getByLabel('Email').fill(ADMIN.email);
  await page.getByLabel('Password').fill(ADMIN.password);
  await page.getByRole('button', { name: 'Sign in' }).click();
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  return { context, page };
}

test('report happy path: photo, GPS, AI check, submit, hazard page', async ({ browser }) => {
  const { page, context } = await newCitizen(browser, { geo: [18.5236, 73.8410] });
  await startReport(page);
  await expect(page.getByText('GPS accuracy ±8 m')).toBeVisible();
  await page.getByRole('button', { name: 'Use this location' }).click();
  await expect(page.getByText(/Looks like a pothole/)).toBeVisible();
  const id = await finishReport(page);
  await expect(page.getByRole('heading', { name: 'Reported. Thank you!' })).toBeVisible();
  await page.goto(`/h/${id}`);
  await expect(page.locator('.badge', { hasText: 'Reported' })).toBeVisible();
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('18.5236, 73.8410');
  await expect(page.getByRole('button', { name: 'Send to city' })).toBeVisible();
  await page.getByRole('button', { name: 'Send to city' }).click();
  await expect(page.getByRole('link', { name: /Open PMC CARE/ })).toBeVisible();
  await context.close();
});

test('report without GPS requires placing the pin', async ({ browser }) => {
  const { page, context } = await newCitizen(browser);
  await startReport(page);
  await expect(page.getByText(/Location is off/)).toBeVisible();
  const blocked = page.getByRole('button', { name: 'Place the pin first' });
  await expect(blocked).toBeDisabled();
  const box = await page.getByTestId('pin-map').boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 90, box.y + box.height / 2 + 40, { steps: 8 });
  await page.mouse.up();
  await page.getByRole('button', { name: 'Use this location' }).click();
  await finishReport(page);
  await context.close();
});

test('second reporter adds a +1 to the nearby hazard', async ({ browser }) => {
  const first = await newCitizen(browser, { geo: [18.5300, 73.8500] });
  await startReport(first.page);
  await first.page.getByRole('button', { name: 'Use this location' }).click();
  const id = await finishReport(first.page);
  await first.context.close();

  const second = await newCitizen(browser, { geo: [18.53005, 73.85004] });
  await startReport(second.page);
  await second.page.getByRole('button', { name: 'Use this location' }).click();
  await expect(second.page.getByText('Already reported nearby')).toBeVisible();
  await second.page.getByRole('button', { name: /That's the one/ }).click();
  await expect(second.page.getByRole('heading', { name: 'Your +1 was added. Thank you!' })).toBeVisible();
  await second.page.goto(`/h/${id}`);
  await expect(second.page.getByText('Reported by 2 people')).toBeVisible();
  await second.context.close();
});

test('officer marks a hazard fixed with a proof photo', async ({ browser }) => {
  const citizen = await newCitizen(browser, { geo: [18.5400, 73.8600] });
  await startReport(citizen.page);
  await citizen.page.getByRole('button', { name: 'Use this location' }).click();
  const id = await finishReport(citizen.page);

  const admin = await adminPage(browser);
  await admin.page.goto('/admin/queue');
  const row = admin.page.getByRole('row').filter({ has: admin.page.getByRole('link', { name: String(id), exact: true }) });
  await row.getByRole('button', { name: 'Update status' }).click();
  await admin.page.getByRole('button', { name: 'Fixed? Confirming' }).click();
  await admin.page.getByLabel('Photo of the repair').setInputFiles(PHOTO);
  await admin.page.getByRole('button', { name: 'Apply' }).click();
  await expect(admin.page.getByRole('dialog')).toBeHidden();

  await citizen.page.goto(`/h/${id}`);
  await expect(citizen.page.getByRole('heading', { name: 'Is it fixed?' })).toBeVisible();
  await citizen.page.getByRole('button', { name: "Yes, it's fixed" }).click();
  await expect(citizen.page.locator('.badge', { hasText: /^Fixed$/ })).toBeVisible();
  await citizen.context.close();
  await admin.context.close();
});

test('moderator rejects a pending report', async ({ browser }) => {
  const citizen = await newCitizen(browser, { geo: [18.5500, 73.8700] });
  await startReport(citizen.page);
  await citizen.page.getByRole('button', { name: 'Use this location' }).click();
  const id = await finishReport(citizen.page, { type: 'Open manhole' });
  await expect(citizen.page.getByText(/A moderator will check it/)).toBeVisible();

  const admin = await adminPage(browser);
  await admin.page.goto('/admin/moderation');
  const card = admin.page.locator('.card').filter({ hasText: `#${id}` });
  await card.getByRole('button', { name: 'Reject: Not a hazard' }).click();
  await expect(card).toHaveCount(0);
  await admin.page.goto(`/h/${id}`);
  await expect(admin.page.getByText('This report was closed by a moderator.')).toBeVisible();
  await citizen.context.close();
  await admin.context.close();
});
