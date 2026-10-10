/**
 * Server-side Firebase ID-token verification with node:crypto only.
 *
 * firebase-admin is not a declared dependency of v1.0 (it is only present
 * transitively in node_modules), so instead of relying on it this module
 * implements Firebase's documented verification procedure
 * (https://firebase.google.com/docs/auth/admin/verify-id-tokens#verify_id_tokens_using_a_third-party_jwt_library):
 *   header: alg RS256, kid present in Google's securetoken public keys
 *   payload: exp in the future, iat and auth_time in the past,
 *            aud == project id, iss == https://securetoken.google.com/<project id>,
 *            sub a non-empty string (<= 128 chars) = the user's uid
 *   signature: RSASSA-PKCS1-v1_5 / SHA-256 with that public key.
 * Keys come from the JWKS endpoint and are cached for the response's
 * Cache-Control max-age. No new npm dependency is required.
 */
import { createPublicKey, verify as cryptoVerify, type JsonWebKey } from 'node:crypto';

export const FIREBASE_JWKS_URL =
  'https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com';

/** clock skew tolerated on iat / auth_time / exp, seconds */
const CLOCK_SKEW_S = 300;

export type Jwks = { keys: Array<JsonWebKey & { kid?: string }> };
export type JwksFetchResult = { jwks: Jwks; maxAgeS: number };
export type JwksFetcher = () => Promise<JwksFetchResult>;

export type VerifiedFirebaseToken = { uid: string; email?: string; claims: Record<string, unknown> };

export class IdTokenError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'IdTokenError';
  }
}

function b64urlDecode(s: string): Buffer {
  return Buffer.from(s.replace(/-/g, '+').replace(/_/g, '/'), 'base64');
}

function parseJson(part: string, what: string): Record<string, unknown> {
  try {
    const v = JSON.parse(b64urlDecode(part).toString('utf8'));
    if (v && typeof v === 'object' && !Array.isArray(v)) return v as Record<string, unknown>;
  } catch {
    /* fall through */
  }
  throw new IdTokenError(`malformed token ${what}`);
}

/** Default fetcher: Google's JWKS with Cache-Control max-age. */
export const fetchGoogleJwks: JwksFetcher = async () => {
  const res = await fetch(FIREBASE_JWKS_URL, { cache: 'no-store' });
  if (!res.ok) throw new IdTokenError(`could not fetch signing keys (HTTP ${res.status})`);
  const jwks = (await res.json()) as Jwks;
  const m = /max-age=(\d+)/.exec(res.headers.get('cache-control') ?? '');
  return { jwks, maxAgeS: m ? Number(m[1]) : 3600 };
};

/** JWKS cache keyed by fetcher, so tests with their own fetcher never share keys. */
const jwksCache = new WeakMap<JwksFetcher, { jwks: Jwks; expiresAtMs: number }>();

async function getJwks(fetcher: JwksFetcher, nowMs: number, forceRefresh = false): Promise<Jwks> {
  const hit = jwksCache.get(fetcher);
  if (hit && !forceRefresh && hit.expiresAtMs > nowMs) return hit.jwks;
  const { jwks, maxAgeS } = await fetcher();
  if (!jwks || !Array.isArray(jwks.keys)) throw new IdTokenError('invalid signing key set');
  jwksCache.set(fetcher, { jwks, expiresAtMs: nowMs + maxAgeS * 1000 });
  return jwks;
}

export async function verifyFirebaseIdToken(
  token: string,
  opts: { projectId: string; fetchJwks?: JwksFetcher; nowMs?: number }
): Promise<VerifiedFirebaseToken> {
  const { projectId } = opts;
  if (!projectId) throw new IdTokenError('Firebase project id is not configured');
  if (typeof token !== 'string' || token.length === 0 || token.length > 4096) {
    throw new IdTokenError('missing or oversized token');
  }
  const parts = token.split('.');
  if (parts.length !== 3) throw new IdTokenError('malformed token');
  const [h, p, sig] = parts;
  const header = parseJson(h, 'header');
  const payload = parseJson(p, 'payload');

  if (header.alg !== 'RS256') throw new IdTokenError('unexpected token algorithm');
  if (typeof header.kid !== 'string' || !header.kid) throw new IdTokenError('token has no key id');

  const nowMs = opts.nowMs ?? Date.now();
  const nowS = Math.floor(nowMs / 1000);
  const exp = payload.exp;
  const iat = payload.iat;
  const authTime = payload.auth_time;
  if (typeof exp !== 'number' || exp + CLOCK_SKEW_S <= nowS) throw new IdTokenError('token expired');
  if (typeof iat !== 'number' || iat - CLOCK_SKEW_S > nowS) throw new IdTokenError('token issued in the future');
  if (typeof authTime !== 'number' || authTime - CLOCK_SKEW_S > nowS) throw new IdTokenError('invalid auth_time');
  if (payload.aud !== projectId) throw new IdTokenError('token audience mismatch');
  if (payload.iss !== `https://securetoken.google.com/${projectId}`) throw new IdTokenError('token issuer mismatch');
  const sub = payload.sub;
  if (typeof sub !== 'string' || sub.length === 0 || sub.length > 128) throw new IdTokenError('invalid subject');

  const fetcher = opts.fetchJwks ?? fetchGoogleJwks;
  let jwks = await getJwks(fetcher, nowMs);
  let jwk = jwks.keys.find((k) => k.kid === header.kid);
  if (!jwk) {
    // key rotation: refresh once
    jwks = await getJwks(fetcher, nowMs, true);
    jwk = jwks.keys.find((k) => k.kid === header.kid);
  }
  if (!jwk) throw new IdTokenError('unknown signing key');

  let ok = false;
  try {
    const key = createPublicKey({ key: jwk, format: 'jwk' });
    ok = cryptoVerify('RSA-SHA256', Buffer.from(`${h}.${p}`), key, b64urlDecode(sig));
  } catch {
    ok = false;
  }
  if (!ok) throw new IdTokenError('invalid token signature');

  return {
    uid: sub,
    ...(typeof payload.email === 'string' ? { email: payload.email } : {}),
    claims: payload,
  };
}
