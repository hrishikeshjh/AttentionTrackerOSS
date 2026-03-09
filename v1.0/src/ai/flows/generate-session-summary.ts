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
  input: {schema: SessionSummaryInputSchema},
  prompt: `You are an AI assistant that generates a session summary report.

  Here are the session metrics:

  Average Attention: {{averageAttention}}%
  Multitask Events: {{multitaskEvents}}
  Engagement Level: {{engagementLevel}}
  Duration: {{duration}}

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
    const {text} = await generateSessionSummaryPrompt(input);

    // Generate PDF
    const pdf = new jsPDF();
    pdf.text('Session Summary Report', 10, 10);
    pdf.text(`Average Attention: ${input.averageAttention}%`, 10, 30);
    pdf.text(`Multitask Events: ${input.multitaskEvents}`, 10, 40);
    pdf.text(`Engagement Level: ${input.engagementLevel}`, 10, 50);
    pdf.text(`Duration: ${input.duration}`, 10, 60);
    pdf.text(`Summary: ${text}`, 10, 80);

    const reportDataUri = pdf.output('datauristring');

    return {reportDataUri};
  }
);
