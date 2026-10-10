/** Tiny argv parser shared by the DACE scripts: --key value, --key=value, repeatable keys, --flag. */
export type Args = { get(k: string): string | undefined; all(k: string): string[]; has(k: string): boolean; positional: string[] };

export function parseArgs(argv: string[]): Args {
  const map = new Map<string, string[]>();
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      positional.push(a);
      continue;
    }
    const eq = a.indexOf('=');
    let key: string;
    let val: string;
    if (eq > 0) {
      key = a.slice(2, eq);
      val = a.slice(eq + 1);
    } else {
      key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith('--')) {
        val = next;
        i++;
      } else val = 'true';
    }
    map.set(key, [...(map.get(key) ?? []), val]);
  }
  return {
    get: (k) => map.get(k)?.at(-1),
    all: (k) => map.get(k) ?? [],
    has: (k) => map.has(k),
    positional,
  };
}

export function die(msg: string): never {
  console.error(msg);
  process.exit(1);
}
