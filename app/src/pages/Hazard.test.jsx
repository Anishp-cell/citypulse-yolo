import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { mockApi, renderPage } from '../test/render';
import Hazard from './Hazard';

const base = {
  id: 7, type: 'pothole', severity: 'high', status: 'open', lat: 18.52, lng: 73.85, road: 'FC Road',
  ward: 'Shivajinagar', city: 'Pune', report_count: 3, days_open: 12, ai_verified: true, priority_score: 60,
  thumb_url: null, first_reported_at: '2026-11-01T10:00:00+00:00', address: 'FC Road, Pune',
  photo_url: '/media/public/p.jpg', photos: ['/media/public/p.jpg'], timeline: [
    { status: 'open', from: null, at: '2026-11-01T10:00:00+00:00', note: 'reported' }],
  official_ticket_id: null, official_portal: null, fixed_at: null, verify_round: 0, is_active: true,
  share_url: 'http://x/h/7', is_following: false, my_vote: null, is_reporter: false,
};

const me = { id: 1, email: 'a@b.in', full_name: 'A', role: 'citizen', language: 'en', city: null, ward: null, created_at: '2026-01-01' };

function show(hazard, user = me) {
  mockApi({ '/api/auth/me': user ? [200, user] : [401, { detail: 'not_authenticated' }], '/api/hazards/7': hazard });
  renderPage(<Hazard />, { route: '/h/7', path: '/h/:id' });
}

describe('Hazard page', () => {
  it('shows the hazard summary and history', async () => {
    show(base);
    expect(await screen.findByText('FC Road')).toBeInTheDocument();
    expect(screen.getByText('Reported by 3 people')).toBeInTheDocument();
    expect(screen.getByText('Open for 12 days')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /send to city/i })).toBeInTheDocument();
  });

  it('asks nearby people to confirm only when a fix is awaiting confirmation', async () => {
    show({ ...base, status: 'fixed_unverified', verify_round: 1 });
    expect(await screen.findByRole('button', { name: "Yes, it's fixed" })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'No, still there' })).toBeInTheDocument();
  });

  it('does not show verify buttons on an open hazard', async () => {
    show(base);
    await screen.findByText('FC Road');
    expect(screen.queryByRole('button', { name: 'No, still there' })).not.toBeInTheDocument();
  });

  it('shows the vote instead of buttons once the user has voted', async () => {
    show({ ...base, status: 'fixed_unverified', verify_round: 1, my_vote: 'fixed' });
    expect(await screen.findByText("You said it's fixed.")).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: "Yes, it's fixed" })).not.toBeInTheDocument();
  });

  it('shows the official complaint number once filed', async () => {
    show({ ...base, status: 'filed', official_ticket_id: 'PMC-55', official_portal: 'PMC CARE' });
    expect(await screen.findByText('Complaint no. PMC-55 (PMC CARE)')).toBeInTheDocument();
  });

  it('shows a not-found message for missing hazards', async () => {
    mockApi({ '/api/auth/me': [401, {}], '/api/hazards/7': [404, { detail: 'hazard_not_found' }] });
    renderPage(<Hazard />, { route: '/h/7', path: '/h/:id' });
    expect(await screen.findByText(/doesn't exist or was removed/)).toBeInTheDocument();
  });
});
