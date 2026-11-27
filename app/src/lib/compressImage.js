// Shrinks phone photos before upload (saves data on mobile networks), fixes orientation and converts
// formats like HEIC to JPEG where the browser can decode them.
import imageCompression from 'browser-image-compression';
// The library's web worker loads its own code with importScripts(), from cdn.jsdelivr.net by default.
// Serve that file from our origin instead: no third-party request, and the CSP can stay 'self'-only.
import workerLibURL from 'browser-image-compression/dist/browser-image-compression.js?url';
import { ApiError } from '../api/client';

const OPTIONS = {
  maxWidthOrHeight: 1600,
  maxSizeMB: 1,
  fileType: 'image/jpeg',
  initialQuality: 0.8,
  useWebWorker: true,
  libURL: new URL(workerLibURL, self.location.href).href, // the worker is a blob: URL, so it needs an absolute URL
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
