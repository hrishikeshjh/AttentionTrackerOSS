/**
 * Pitch characterisation of the LEGACY head-pose formula (proposal section 1,
 * annex E). Applies the shipped calculateHeadPose() and the shipped MLEngine
 * pitch sub-score to the standard OpenCV 3D head model under known rotations
 * and checks the proposal's table. It also emits the constants actually used
 * by the code to docs/proposals/evidence/legacy_pitch_constants.json, which
 * docs/proposals/evidence/pitch_sim.py reads, so the Python reproduction can
 * never drift from the TypeScript source.
 *
 * This documents a known defect (pitch rises when the head tilts down); it is
 * a characterisation test, not a statement that the behaviour is correct.
 */
import { describe, it, expect } from 'vitest';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { calculateHeadPose } from '../pose/head-pose';
import { HEAD_POSE_CALIBRATION, HEAD_POSE_LANDMARKS, OPTIMAL_RANGES, FEATURE_WEIGHTS_LEGACY } from '../constants';
import { MLEngine } from '../../ml-engine';
import type { Point2D } from '../geometry';
import type { ExtendedFaceMetrics } from '../../types';

/** Standard OpenCV head model (mm; x right, y up, z toward the viewer). */
const MODEL = {
  noseTip: [0, 0, 0],
  chin: [0, -330, -65],
  leftEyeLeftCorner: [-225, 170, -135],
  rightEyeRightCorner: [225, 170, -135],
  leftMouthCorner: [-150, -150, -125],
  rightMouthCorner: [150, -150, -125],
} as const;

/**
 * Rotate about the horizontal axis through the nose tip by `deg` (+ = head
 * down) and project to the image (y down). Orthographic projection (camera
 * far away) is what the proposal's table assumes; `distanceMm` adds a
 * pinhole camera at that distance (f = 640 px).
 */
function project(p: readonly number[], deg: number, distanceMm: number | null): Point2D {
  // camera frame: x right, y down, z away from the camera; model -> (x, -y, -z)
  const x = p[0], y = -p[1], z = -p[2];
  const a = (deg * Math.PI) / 180;
  const yr = Math.cos(a) * y - Math.sin(a) * z;
  const zr = Math.sin(a) * y + Math.cos(a) * z;
  if (distanceMm === null) return { x: 320 + x * 0.2, y: 240 + yr * 0.2 };
  const Z = distanceMm + zr;
  return { x: 320 + (640 * x) / Z, y: 240 + (640 * yr) / Z };
}

function landmarksFor(deg: number, distanceMm: number | null): Point2D[] {
  const lm: Point2D[] = Array.from({ length: 68 }, () => ({ x: 320, y: 240 }));
  lm[HEAD_POSE_LANDMARKS.nose] = project(MODEL.noseTip, deg, distanceMm);
  lm[HEAD_POSE_LANDMARKS.chin] = project(MODEL.chin, deg, distanceMm);
  lm[HEAD_POSE_LANDMARKS.leftEyeCorner] = project(MODEL.leftEyeLeftCorner, deg, distanceMm);
  lm[HEAD_POSE_LANDMARKS.rightEyeCorner] = project(MODEL.rightEyeRightCorner, deg, distanceMm);
  return lm;
}

/** Pitch sub-score as computed by the shipped MLEngine (legacy weights path). */
function legacyPitchSubScore(pitch: number): number {
  const m: ExtendedFaceMetrics = {
    ear: null, mar: null, yaw: null, pitch, roll: null, blinkDetected: false, blinkRate: 0, perclos: 0,
    gazeX: null, gazeY: null, faceDistance: null, faceArea: null, fatigueIndex: 0, fidgetScore: 0,
    headPoseStability: 100, eyebrowRaise: null, mouthOpenness: null, earVariance: 0, marVariance: 0, poseVariance: 0,
  };
  const c = new MLEngine().predict(m).featureContributions.pitchDeviation;
  return c / FEATURE_WEIGHTS_LEGACY.pitchDeviation + 50;
}

const TILTS = [-30, 0, 15, 30, 45]; // + = head down
const PROPOSAL = {
  pitch: [5.9, 12.6, 14.7, 15.9, 16.0],
  subScore: [92.1, 83.2, 80.4, 78.2, 78.1],
};

describe('legacy pitch characterisation (proposal section 1)', () => {
  const rows = TILTS.map((deg) => {
    const pitch = calculateHeadPose(landmarksFor(deg, null)).pitch as number;
    return { trueTiltDeg: deg, legacyPitch: pitch, subScore: legacyPitchSubScore(pitch) };
  });

  it('reproduces the proposal table (orthographic projection)', () => {
    rows.forEach((r, i) => {
      expect(Math.abs(r.legacyPitch - PROPOSAL.pitch[i])).toBeLessThanOrEqual(0.05 + 1e-9);
      expect(Math.abs(r.subScore - PROPOSAL.subScore[i])).toBeLessThanOrEqual(0.1 + 1e-9);
    });
  });

  it('pitch rises (never goes below −15) when the head tilts down, so the reading/phone rule cannot fire', () => {
    for (let i = 1; i < rows.length; i++) expect(rows[i].legacyPitch).toBeGreaterThan(rows[i - 1].legacyPitch);
    for (const r of rows) expect(r.legacyPitch).toBeGreaterThan(-15);
    // looking up scores better than looking at the screen
    expect(rows[0].subScore).toBeGreaterThan(rows[1].subScore);
  });

  it('the pitch sub-score curve is the documented one', () => {
    const max = OPTIMAL_RANGES.yaw.maxDeviation;
    for (const v of [0, 5.9, 12.6, 15, 15.9, 16, 40, 80]) {
      const expected = v <= max ? 100 - (v / max) * 20 : Math.max(30, 80 - (v - max) * 2);
      expect(legacyPitchSubScore(v)).toBeCloseTo(expected, 9);
    }
  });

  it('emits the constants JSON consumed by pitch_sim.py', () => {
    const here = dirname(fileURLToPath(import.meta.url));
    const repoRoot = resolve(here, '../../../../..');
    const out = resolve(repoRoot, 'docs/proposals/evidence/legacy_pitch_constants.json');
    const payload = {
      _comment:
        'GENERATED by v1.0/src/lib/scoring/__tests__/pitch-characterisation.test.ts from the shipped TypeScript constants. Do not edit by hand.',
      sources: [
        'v1.0/src/lib/scoring/pose/head-pose.ts',
        'v1.0/src/lib/scoring/constants.ts',
        'v1.0/src/lib/ml-engine.ts (scoreFeature yawDeviation/pitchDeviation branch; computeContextAwarePitch)',
      ],
      formula: 'pitch = (nose.y - eyeMidpoint.y) / interocularDistance * PITCH_GAIN (image y down)',
      landmarks: HEAD_POSE_LANDMARKS,
      PITCH_GAIN: HEAD_POSE_CALIBRATION.PITCH_GAIN,
      YAW_GAIN: HEAD_POSE_CALIBRATION.YAW_GAIN,
      pitchSubScore: {
        maxDeviation: OPTIMAL_RANGES.yaw.maxDeviation,
        inside: '100 - |v| / maxDeviation * 20',
        outside: 'max(30, 80 - (|v| - maxDeviation) * 2)',
        constants: { top: 100, insideDrop: 20, outsideStart: 80, outsideSlope: 2, floor: 30 },
      },
      contextAwarePitchDownThreshold: -15,
      model3dMm: MODEL,
      tiltsDeg: TILTS,
      proposalTable: PROPOSAL,
      typescriptResults: rows.map((r) => ({ ...r, legacyPitch: +r.legacyPitch.toFixed(6), subScore: +r.subScore.toFixed(6) })),
    };
    const text = JSON.stringify(payload, null, 2) + '\n';
    if (!existsSync(dirname(out))) mkdirSync(dirname(out), { recursive: true });
    if (!existsSync(out) || readFileSync(out, 'utf8') !== text) writeFileSync(out, text);
    expect(JSON.parse(readFileSync(out, 'utf8')).PITCH_GAIN).toBe(HEAD_POSE_CALIBRATION.PITCH_GAIN);
  });
});
