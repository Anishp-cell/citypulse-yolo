// End-to-end tests against the real backend (real ONNX model, temporary SQLite DB, geocoding off).
// Run: npm run build && npx playwright test
import { defineConfig, devices } from '@playwright/test';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '..');
const runDir = resolve(import.meta.dirname, '.e2e-runs', `run-${Date.now()}`);
mkdirSync(runDir, { recursive: true });
const python = process.env.E2E_PYTHON || (process.platform === 'win32' ? resolve(root, 'venv/Scripts/python.exe') : 'python');
const PORT = 8123;
export const ADMIN = { email: 'admin@e2e.test', password: 'admin-pass-123' };

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    serviceWorkers: 'block',
    trace: 'retain-on-failure',
    ...devices['Pixel 7'],
  },
  projects: [{ name: 'mobile-chromium', use: { browserName: 'chromium' } }],
  webServer: {
    command: [
      `"${python}" -m alembic upgrade head`,
      `"${python}" -m backend.cli create-admin --email ${ADMIN.email} --password ${ADMIN.password}`,
      `"${python}" -m uvicorn backend.main:create_app --factory --host 127.0.0.1 --port ${PORT}`,
    ].join(' && '),
    cwd: root,
    url: `http://127.0.0.1:${PORT}/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: {
      APP_ENV: 'test',
      SECRET_KEY: 'e2e-secret-key-0123456789abcdef0123',
      DATABASE_URL: `sqlite:///${runDir.replace(/\\/g, '/')}/e2e.db`,
      MEDIA_DIR: resolve(runDir, 'media'),
      FRONTEND_DIST: resolve(import.meta.dirname, 'dist'),
      GEOCODER: 'off',
      RATE_LIMITS_ENABLED: 'false',
      DETECTOR: 'onnx',
      APP_URL: `http://127.0.0.1:${PORT}`,
    },
  },
});
