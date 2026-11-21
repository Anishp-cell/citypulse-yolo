import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import { HAZARD_TYPES, STATUS } from './status';

const backend = (p) => readFileSync(join(__dirname, '../../../backend', p), 'utf8');

describe('status metadata matches the backend', () => {
  it('has exactly the backend HazardStatus values', () => {
    const enumBody = backend('domain/status.py').split('class HazardStatus')[1].split('\n\n')[0];
    const values = [...enumBody.matchAll(/^\s+(\w+) = "(\w+)"/gm)].map((m) => m[2]);
    expect(values.length).toBeGreaterThan(5);
    expect(Object.keys(STATUS).sort()).toEqual(values.sort());
  });

  it('has exactly the backend hazard types, in order', () => {
    const m = backend('models.py').match(/HAZARD_TYPES = \(([^)]*)\)/);
    const values = [...m[1].matchAll(/"(\w+)"/g)].map((x) => x[1]);
    expect(HAZARD_TYPES.map((t) => t.value)).toEqual(values);
  });
});
