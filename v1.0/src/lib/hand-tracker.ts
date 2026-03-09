'use client';

/**
 * HandTracker — MediaPipe Hands wrapper for browser
 *
 * Detects up to 2 hands and returns 21 normalized landmarks per hand.
 * Results are consumed by HandObjectProximityDetector in face-geometry.ts
 * to compute the hand-object proximity signal (ARCHITECTURE.md §2.4).
 *
 * Throttled to every 3rd frame (same cadence as ObjectDetector).
 *
 * New dependency: @mediapipe/hands
 *   npm install @mediapipe/hands
 *   WASM/model files are loaded from jsDelivr CDN at runtime.
 */

import type { HandResult, HandLandmark } from './types';

// ==================== HAND TRACKER ====================

export class HandTracker {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private hands: any = null;
  private frameCounter = 0;
  private lastResults: HandResult[] = [];

  /**
   * Load MediaPipe Hands (Lite model, complexity 0).
   * Safe to call multiple times; re-initializes on each call.
   */
  async initialize(): Promise<void> {
    if (typeof window === 'undefined') return; // SSR guard

    try {
      // Indirect import so TS resolves as Promise<any> before package is installed.
      // Install: npm install @mediapipe/hands
      const pkg = '@mediapipe/hands';
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { Hands } = await import(pkg as string) as any;

      const hands = new Hands({
        locateFile: (file: string) =>
          `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`,
      });

      hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 0,          // Lite model — ~15 ms/frame
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      hands.onResults((results: unknown) => {
        this.lastResults = HandTracker.parseResults(results);
      });

      this.hands = hands;
    } catch {
      console.warn(
        '[HandTracker] @mediapipe/hands failed to load. Hand tracking disabled.'
      );
    }
  }

  isLoaded(): boolean {
    return this.hands !== null;
  }

  /**
   * Run hand detection on a video frame.
   * Throttled: skips 2 out of every 3 frames, returns cached results.
   */
  async detect(source: HTMLVideoElement): Promise<HandResult[]> {
    this.frameCounter++;
    if (this.frameCounter % 3 !== 0) return this.lastResults;
    if (!this.hands) return [];

    await this.hands.send({ image: source });
    return this.lastResults;
  }

  // ==================== STATIC HELPERS ====================

  /**
   * Compute the centroid of a hand (average of all 21 landmarks).
   * Coordinates are normalized [0, 1] matching the source video dimensions.
   */
  static getHandCenter(landmarks: HandLandmark[]): { x: number; y: number } {
    const sum = landmarks.reduce(
      (acc, lm) => ({ x: acc.x + lm.x, y: acc.y + lm.y }),
      { x: 0, y: 0 }
    );
    return {
      x: sum.x / landmarks.length,
      y: sum.y / landmarks.length,
    };
  }

  /** Euclidean distance between two normalized 2D points. */
  static distance2D(
    a: { x: number; y: number },
    b: { x: number; y: number }
  ): number {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
  }

  // ==================== PRIVATE ====================

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private static parseResults(results: any): HandResult[] {
    if (
      !results.multiHandLandmarks ||
      results.multiHandLandmarks.length === 0
    ) {
      return [];
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    return results.multiHandLandmarks.map((landmarks: any[], idx: number) => ({
      landmarks: landmarks.map(
        (lm: { x: number; y: number; z: number }) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z,
        })
      ),
      handedness:
        (results.multiHandedness?.[idx]?.label as 'Left' | 'Right') ?? 'Left',
    }));
  }
}
