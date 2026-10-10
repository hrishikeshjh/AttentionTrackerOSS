/**
 * Core of the report server action (src/app/actions.ts), kept free of
 * Next.js / Genkit imports so it can be unit-tested with injected
 * dependencies. Order: authenticate (Firebase ID token) -> rate limit per uid
 * -> validate input -> generate. Errors returned to the client are generic;
 * details are logged server-side only.
 */
import { ReportInputSchema, type ReportInput } from './report-input';
import { IdTokenError, type VerifiedFirebaseToken } from './firebase-id-token';
import type { SlidingWindowRateLimiter } from './rate-limit';

export type ReportActionResult =
  | { success: true; data: string }
  | { success: false; error: string; code: 'unauthenticated' | 'rate_limited' | 'invalid_input' | 'server_error' };

export type ReportDeps = {
  verifyToken: (token: string) => Promise<VerifiedFirebaseToken>;
  limiter: SlidingWindowRateLimiter;
  generate: (input: ReportInput) => Promise<string>;
  log?: (msg: string, err?: unknown) => void;
};

export async function handleReportRequest(
  idToken: unknown,
  rawInput: unknown,
  deps: ReportDeps
): Promise<ReportActionResult> {
  const log = deps.log ?? ((m, e) => console.error(m, e));

  let user: VerifiedFirebaseToken;
  try {
    if (typeof idToken !== 'string' || !idToken) throw new IdTokenError('missing token');
    user = await deps.verifyToken(idToken);
  } catch (e) {
    log('[report] authentication failed:', e instanceof Error ? e.message : e);
    return { success: false, code: 'unauthenticated', error: 'Please sign in again to export a report.' };
  }

  const rl = deps.limiter.check(user.uid);
  if (!rl.allowed) {
    const mins = Math.max(1, Math.ceil(rl.retryAfterMs / 60000));
    return { success: false, code: 'rate_limited', error: `Report limit reached. Try again in about ${mins} min.` };
  }

  const parsed = ReportInputSchema.safeParse(rawInput);
  if (!parsed.success) {
    log('[report] invalid input:', parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; '));
    return { success: false, code: 'invalid_input', error: 'The session summary could not be validated.' };
  }

  try {
    const data = await deps.generate(parsed.data);
    return { success: true, data };
  } catch (e) {
    log('[report] generation failed:', e);
    return { success: false, code: 'server_error', error: 'Failed to generate report.' };
  }
}
