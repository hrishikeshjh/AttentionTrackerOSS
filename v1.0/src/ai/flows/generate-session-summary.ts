'use server';

/**
 * @fileOverview This file defines a Genkit flow for generating a session summary report.
 *
 * - generateSessionSummary - A function that generates a session summary report.
 * - SessionSummaryInput - The input type for the generateSessionSummary function.
 * - SessionSummaryOutput - The return type for the generateSessionSummary function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import {jsPDF} from 'jspdf';

const SessionSummaryInputSchema = z.object({
  averageAttention: z
    .number()
    .describe('The average attention score for the session.'),
  multitaskEvents: z
    .number()
    .describe('The number of multitasking events during the session.'),
  engagementLevel: z
    .string()
    .describe('The overall engagement level (high, moderate, low).'),
  duration: z.string().describe('The session duration (e.g., 15m).'),
  // DACE (optional; only sent when the session carries "Unsure" flags)
  unsureShare: z
    .number()
    .min(0)
    .max(1)
    .optional()
    .describe('Share (0-1) of the session for which the activity could not be determined. Never a penalty.'),
  unsureReasons: z
    .string()
    .max(300)
    .optional()
    .describe('Human-readable reasons with counts, e.g. "desk not visible (12), poor tracking (2)".'),
});

export type SessionSummaryInput = z.infer<typeof SessionSummaryInputSchema>;

const SessionSummaryOutputSchema = z.object({
  reportDataUri: z
    .string()
    .describe(
      'The session summary report as a data URI that must include a MIME type and use Base64 encoding. Expected format: \'data:<mimetype>;base64,<encoded_data>\'.' 
    ),
});

export type SessionSummaryOutput = z.infer<typeof SessionSummaryOutputSchema>;

export async function generateSessionSummary(
  input: SessionSummaryInput
): Promise<SessionSummaryOutput> {
  return generateSessionSummaryFlow(input);
}

const generateSessionSummaryPrompt = ai.definePrompt({
  name: 'generateSessionSummaryPrompt',
  input: {schema: SessionSummaryInputSchema.extend({unsurePercent: z.number().optional()})},
  prompt: `You are an AI assistant that generates a session summary report.

  Here are the session metrics:

  Average Attention: {{averageAttention}}%
  Multitask Events: {{multitaskEvents}}
  Engagement Level: {{engagementLevel}}
  Duration: {{duration}}
  {{#if unsurePercent}}
  Undetermined activity: CognEn could not determine the activity for {{unsurePercent}}% of the session{{#if unsureReasons}} (reasons: {{unsureReasons}}){{/if}}. This time is excluded from the average and is not a penalty; mention it neutrally in one sentence.
  {{/if}}

  Please generate a concise summary of the session in a professional tone.
  `,
});

const generateSessionSummaryFlow = ai.defineFlow(
  {
    name: 'generateSessionSummaryFlow',
    inputSchema: SessionSummaryInputSchema,
    outputSchema: SessionSummaryOutputSchema,
  },
  async input => {
    const unsurePercent =
      input.unsureShare !== undefined && input.unsureShare > 0 ? Math.round(input.unsureShare * 100) : undefined;
    const {text} = await generateSessionSummaryPrompt({...input, unsurePercent});

    // Generate PDF
    const pdf = new jsPDF();
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    const maxWidth = pageWidth - 20;
    const lineHeight = 7;
    pdf.text('Session Summary Report', 10, 10);
    pdf.text(`Average Attention: ${input.averageAttention}%`, 10, 30);
    pdf.text(`Multitask Events: ${input.multitaskEvents}`, 10, 40);
    pdf.text(`Engagement Level: ${input.engagementLevel}`, 10, 50);
    pdf.text(`Duration: ${input.duration}`, 10, 60);
    let y = 80;
    if (unsurePercent !== undefined) {
      const unsureLine =
        `Undetermined: CognEn could not determine the activity for ${unsurePercent}% of the session` +
        (input.unsureReasons ? ` (${input.unsureReasons})` : '') +
        '. Excluded from the average; not a penalty.';
      const lines = pdf.splitTextToSize(unsureLine, maxWidth) as string[];
      pdf.text(lines, 10, 70);
      y = Math.max(y, 70 + lines.length * lineHeight + 3);
    }
    // Wrap the summary and continue on new pages instead of running off the page.
    const summaryLines = pdf.splitTextToSize(`Summary: ${text}`, maxWidth) as string[];
    for (const line of summaryLines) {
      if (y > pageHeight - 10) {
        pdf.addPage();
        y = 15;
      }
      pdf.text(line, 10, y);
      y += lineHeight;
    }

    const reportDataUri = pdf.output('datauristring');

    return {reportDataUri};
  }
);
