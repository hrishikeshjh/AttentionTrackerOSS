'use server';

/**
 * Report export server action.
 *
 * Server actions are public POST endpoints, so this one authenticates the
 * caller itself: the client sends its Firebase ID token
 * (auth.currentUser.getIdToken()), which is verified here against Google's
 * public keys (src/lib/server/firebase-id-token.ts, node:crypto only), then
 * rate-limited per user and strictly validated before Gemini is called.
 *
 * Why not middleware: the dashboard's auth guard (app/dashboard/layout.tsx)
 * is client-side because Firebase client auth lives in the browser
 * (IndexedDB), and Next.js middleware cannot read it without a session
 * cookie. The guard only hides UI; the protected server-side resource is this
 * action (the dashboard page itself holds no server data), and it is now
 * enforced here.
 */
import { generateSessionSummary } from '@/ai/flows/generate-session-summary';
import { verifyFirebaseIdToken } from '@/lib/server/firebase-id-token';
import { SlidingWindowRateLimiter } from '@/lib/server/rate-limit';
import { handleReportRequest, type ReportActionResult } from '@/lib/server/report-request';

/** 5 reports per user per 10 minutes (per server instance, see rate-limit.ts). */
const limiter = new SlidingWindowRateLimiter(5, 10 * 60 * 1000);

export async function generateReportAction(idToken: string, input: unknown): Promise<ReportActionResult> {
  const projectId = process.env.FIREBASE_PROJECT_ID || process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || '';
  return handleReportRequest(idToken, input, {
    verifyToken: (token) => verifyFirebaseIdToken(token, { projectId }),
    limiter,
    generate: async (valid) => (await generateSessionSummary(valid)).reportDataUri,
  });
}
