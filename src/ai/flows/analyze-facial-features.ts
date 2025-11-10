'use server';

/**
 * @fileOverview Analyzes facial features from webcam feed to determine attention state.
 *
 * - analyzeFacialFeatures - A function that handles the facial feature analysis process.
 * - AnalyzeFacialFeaturesInput - The input type for the analyzeFacialFeatures function.
 * - AnalyzeFacialFeaturesOutput - The return type for the analyzeFacialFeatures function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AnalyzeFacialFeaturesInputSchema = z.object({
  faceLandmarks: z.string().describe('The face landmarks detected by Mediapipe FaceMesh, as a JSON string.'),
  ear: z.number().describe('Eye Aspect Ratio (EAR) calculated from face landmarks.'),
  mar: z.number().describe('Mouth Aspect Ratio (MAR) calculated from face landmarks.'),
  yaw: z.number().describe('Head pose yaw.'),
  pitch: z.number().describe('Head pose pitch.'),
  roll: z.number().describe('Head pose roll.'),
});
export type AnalyzeFacialFeaturesInput = z.infer<typeof AnalyzeFacialFeaturesInputSchema>;

const AnalyzeFacialFeaturesOutputSchema = z.object({
  attentionState: z.enum(['Attentive', 'Distracted', 'Multitasking']).describe('The determined attention state of the user.'),
});
export type AnalyzeFacialFeaturesOutput = z.infer<typeof AnalyzeFacialFeaturesOutputSchema>;

export async function analyzeFacialFeatures(input: AnalyzeFacialFeaturesInput): Promise<AnalyzeFacialFeaturesOutput> {
  return analyzeFacialFeaturesFlow(input);
}

const prompt = ai.definePrompt({
  name: 'analyzeFacialFeaturesPrompt',
  input: {schema: AnalyzeFacialFeaturesInputSchema},
  output: {schema: AnalyzeFacialFeaturesOutputSchema},
  prompt: `Given the following facial feature data, determine the attention state of the user. The possible states are Attentive, Distracted, and Multitasking.\n\nFace Landmarks: {{{faceLandmarks}}}\nEye Aspect Ratio (EAR): {{{ear}}}\nMouth Aspect Ratio (MAR): {{{mar}}}\nHead Pose (Yaw): {{{yaw}}}\nHead Pose (Pitch): {{{pitch}}}\nHead Pose (Roll): {{{roll}}}\n\nConsider that high EAR and MAR values typically indicate attentiveness. Significant head pose variations or low EAR/MAR may indicate distraction or multitasking.\n\nOutput the attention state.`,
});

const analyzeFacialFeaturesFlow = ai.defineFlow(
  {
    name: 'analyzeFacialFeaturesFlow',
    inputSchema: AnalyzeFacialFeaturesInputSchema,
    outputSchema: AnalyzeFacialFeaturesOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
