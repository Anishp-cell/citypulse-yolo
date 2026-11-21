// Shrinks phone photos before upload (saves data on mobile networks), fixes orientation and converts
// formats like HEIC to JPEG where the browser can decode them.
import imageCompression from 'browser-image-compression';
import { ApiError } from '../api/client';

const OPTIONS = {
  maxWidthOrHeight: 1600,
  maxSizeMB: 1,
  fileType: 'image/jpeg',
  initialQuality: 0.8,
  useWebWorker: true,
};

export async function compressImage(file) {
  if (file.type && !file.type.startsWith('image/')) throw new ApiError(415, 'unsupported_image');
  let blob;
  try {
    blob = await imageCompression(file, OPTIONS);
  } catch {
    throw new ApiError(415, 'unsupported_image');
  }
  const base = (file.name || 'photo').replace(/\.[^.]+$/, '');
  return new File([blob], `${base}.jpg`, { type: 'image/jpeg' });
}
