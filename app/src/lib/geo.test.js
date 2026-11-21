import { describe, expect, it } from 'vitest';
import { bboxParam, haversineM } from './geo';

describe('geo', () => {
  it('computes distances', () => {
    const d = haversineM(18.5195, 73.8553, 18.5289, 73.8744);
    expect(d).toBeGreaterThan(2200);
    expect(d).toBeLessThan(2400);
  });

  it('builds a minLng,minLat,maxLng,maxLat bbox around a point', () => {
    const [minLng, minLat, maxLng, maxLat] = bboxParam(18.52, 73.85, 1000).split(',').map(Number);
    expect(minLat).toBeLessThan(18.52);
    expect(maxLat).toBeGreaterThan(18.52);
    expect(minLng).toBeLessThan(73.85);
    expect(maxLng).toBeGreaterThan(73.85);
    expect(haversineM(18.52, 73.85, maxLat, 73.85)).toBeGreaterThanOrEqual(999);
  });
});
