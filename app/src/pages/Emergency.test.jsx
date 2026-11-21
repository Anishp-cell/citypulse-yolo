import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { renderPage } from '../test/render';
import Emergency from './Emergency';

describe('Emergency page', () => {
  it('works with no network at all and offers 112 first', async () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'));
    renderPage(<Emergency />);
    const call = screen.getByRole('link', { name: /112/ });
    expect(call).toHaveAttribute('href', 'tel:112');
    expect(screen.getByRole('link', { name: /108/ })).toHaveAttribute('href', 'tel:108');
  });

  it('shows the steps for the chosen situation', () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'));
    renderPage(<Emergency />);
    fireEvent.click(screen.getByRole('button', { name: 'Someone is injured' }));
    expect(screen.getByText(/Do not move an injured person/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Only vehicles are damaged' }));
    expect(screen.getByText(/Exchange names, phone numbers/)).toBeInTheDocument();
  });

  it('always shows the disclaimer', () => {
    vi.spyOn(globalThis, 'fetch').mockRejectedValue(new TypeError('offline'));
    renderPage(<Emergency />);
    expect(screen.getByText(/not medical advice/)).toBeInTheDocument();
  });
});
