'use server';

import { generateSessionSummary, type SessionSummaryInput } from '@/ai/flows/generate-session-summary';

export async function generateReportAction(input: SessionSummaryInput) {
  try {
    const { reportDataUri } = await generateSessionSummary(input);
    return { success: true, data: reportDataUri };
  } catch (error) {
    console.error('Error in generateReportAction:', error);
    const errorMessage = error instanceof Error ? error.message : 'An unexpected error occurred.';
    return { success: false, error: `Failed to generate report: ${errorMessage}` };
  }
}
