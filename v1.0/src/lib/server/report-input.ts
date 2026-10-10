/**
 * Strict validation of the report server action's input. The Genkit flow's
 * own schema only checks types; values from the client are untrusted, are
 * interpolated into the Gemini prompt and printed in the PDF, so ranges,
 * lengths and allowed characters are enforced here and unknown keys rejected.
 */
import { z } from 'zod';

const SAFE_TEXT = /^[\p{L}\p{N} .,:;()%'/+-]*$/u;

export const ReportInputSchema = z
  .object({
    averageAttention: z.number().finite().min(0).max(100),
    multitaskEvents: z.number().int().min(0).max(100_000),
    engagementLevel: z.enum(['High', 'Moderate', 'Low']),
    duration: z.string().max(32).regex(/^[0-9hms :]+$/, 'invalid duration'),
    unsureShare: z.number().finite().min(0).max(1).optional(),
    unsureReasons: z.string().max(300).regex(SAFE_TEXT, 'invalid characters').optional(),
  })
  .strict();

export type ReportInput = z.infer<typeof ReportInputSchema>;
