import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import en from './en.json';
import hi from './hi.json';
import mr from './mr.json';

function flatten(obj, prefix = '') {
  return Object.entries(obj).flatMap(([k, v]) => {
    const key = prefix ? `${prefix}.${k}` : k;
    if (Array.isArray(v)) return [`${key}[${v.length}]`];
    return v && typeof v === 'object' ? flatten(v, key) : [key];
  });
}

function sourceFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return sourceFiles(p);
    return /\.(jsx?)$/.test(name) && !name.includes('.test.') ? [p] : [];
  });
}

const enKeys = new Set(flatten(en).map((k) => k.replace(/\[\d+\]$/, '')));
const plural = (k) => enKeys.has(`${k}_one`) && enKeys.has(`${k}_other`);

describe('translations', () => {
  it.each([['hi', hi], ['mr', mr]])('%s has exactly the English keys (and list lengths)', (_, dict) => {
    expect(flatten(dict).sort()).toEqual(flatten(en).sort());
  });

  it('every literal t() key used in the source exists in en.json', () => {
    const missing = [];
    for (const file of sourceFiles(join(__dirname, '..'))) {
      const src = readFileSync(file, 'utf8');
      for (const m of src.matchAll(/\bt\(\s*['"]([a-z0-9_.]+)['"]/g)) {
        if (!enKeys.has(m[1]) && !plural(m[1])) missing.push(`${file}: ${m[1]}`);
      }
    }
    expect(missing).toEqual([]);
  });
});
