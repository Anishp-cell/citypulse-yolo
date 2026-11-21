import { describe, expect, it, vi } from 'vitest';
import imageCompression from 'browser-image-compression';
import { compressImage } from './compressImage';

vi.mock('browser-image-compression', () => ({ default: vi.fn() }));

describe('compressImage', () => {
  it('resizes to 1600px JPEG and names the file .jpg', async () => {
    imageCompression.mockResolvedValue(new Blob(['x'], { type: 'image/jpeg' }));
    const out = await compressImage(new File(['raw'], 'IMG_0001.HEIC', { type: 'image/heic' }));
    const opts = imageCompression.mock.calls[0][1];
    expect(opts).toMatchObject({ maxWidthOrHeight: 1600, fileType: 'image/jpeg', initialQuality: 0.8 });
    expect(out.name).toBe('IMG_0001.jpg');
    expect(out.type).toBe('image/jpeg');
  });

  it('turns decode failures into an unsupported_image error', async () => {
    imageCompression.mockRejectedValue(new Error('cannot decode'));
    await expect(compressImage(new File(['raw'], 'x.heic'))).rejects.toMatchObject({ code: 'unsupported_image' });
  });

  it('rejects non-image files', async () => {
    await expect(compressImage(new File(['raw'], 'doc.pdf', { type: 'application/pdf' })))
      .rejects.toMatchObject({ code: 'unsupported_image' });
  });
});
