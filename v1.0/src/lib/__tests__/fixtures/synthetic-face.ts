/**
 * Deterministic synthetic inputs for the legacy-equivalence (golden) test.
 *
 * Produces face-api-style 68-point landmark arrays (pixel coordinates),
 * MediaPipe-style hand results and detector outputs from a seeded PRNG, so
 * the same sequence is generated on every run and every machine. The shapes
 * are not anatomically exact; they only need to drive every legacy code path
 * (EAR/MAR, head pose, gaze, blink/PERCLOS, fatigue, proximity, temporal
 * rules, context-aware scoring) deterministically.
 */
import type { Point2D } from '../../scoring/geometry';
import type { DetectedObject, HandResult, ObjectClass } from '../../types';

/** mulberry32: small deterministic PRNG. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export type FaceParams = {
  /** eye openness multiplier (1 = open, 0 = fully closed) */
  eyeOpen: number;
  /** horizontal nose/mouth shift as a fraction of face half-width (yaw proxy) */
  yawShift: number;
  /** vertical nose shift in template units (pitch proxy; negative = nose up) */
  noseShiftY: number;
  /** in-plane rotation in degrees */
  rollDeg: number;
  /** mouth opening multiplier */
  mouthOpen: number;
  /** horizontal iris-centroid bias (moves eyelid mid points) */
  lidShiftX: number;
  /** face scale in px per template unit */
  scale: number;
  cx: number;
  cy: number;
  /** pixel noise amplitude */
  noise: number;
};

export const DEFAULT_FACE: FaceParams = {
  eyeOpen: 1,
  yawShift: 0,
  noseShiftY: 0,
  rollDeg: 0,
  mouthOpen: 0.3,
  lidShiftX: 0,
  scale: 100,
  cx: 320,
  cy: 240,
  noise: 0.6,
};

/** Template 68-point face in a unit frame (y down). */
function template(p: FaceParams): Point2D[] {
  const pts: Point2D[] = [];
  // Jaw 0..16
  for (let i = 0; i <= 16; i++) {
    const th = (Math.PI * i) / 16;
    pts.push({ x: -Math.cos(th), y: -0.2 + 1.2 * Math.sin(th) });
  }
  // Right brow 17..21 (image left), left brow 22..26
  for (let i = 0; i < 5; i++) pts.push({ x: -0.8 + i * 0.16, y: -0.55 - 0.05 * Math.sin((Math.PI * i) / 4) });
  for (let i = 0; i < 5; i++) pts.push({ x: 0.16 + i * 0.16, y: -0.55 - 0.05 * Math.sin((Math.PI * (4 - i)) / 4) });
  // Nose bridge 27..30
  const ny = p.noseShiftY;
  const nx = p.yawShift;
  pts.push({ x: nx * 0.4, y: -0.35 });
  pts.push({ x: nx * 0.6, y: -0.2 + ny * 0.5 });
  pts.push({ x: nx * 0.8, y: -0.05 + ny * 0.8 });
  pts.push({ x: nx, y: 0.1 + ny });
  // Nostrils 31..35
  for (let i = 0; i < 5; i++) pts.push({ x: nx + (-0.2 + i * 0.1), y: 0.2 + ny * 0.8 + (i === 2 ? 0.02 : 0) });
  // Eyes: lid half-height h = 0.045 open -> EAR = 4h / (2 * 0.3) = 0.3
  const h = 0.045 * p.eyeOpen;
  const lx = p.lidShiftX;
  // Right eye 36..41
  pts.push({ x: -0.6, y: -0.3 });
  pts.push({ x: -0.5 + lx, y: -0.3 - h });
  pts.push({ x: -0.4 + lx, y: -0.3 - h });
  pts.push({ x: -0.3, y: -0.3 });
  pts.push({ x: -0.4 + lx, y: -0.3 + h });
  pts.push({ x: -0.5 + lx, y: -0.3 + h });
  // Left eye 42..47
  pts.push({ x: 0.3, y: -0.3 });
  pts.push({ x: 0.4 + lx, y: -0.3 - h });
  pts.push({ x: 0.5 + lx, y: -0.3 - h });
  pts.push({ x: 0.6, y: -0.3 });
  pts.push({ x: 0.5 + lx, y: -0.3 + h });
  pts.push({ x: 0.4 + lx, y: -0.3 + h });
  // Outer mouth 48..59 (ellipse), inner mouth 60..67
  const mcx = nx * 0.9;
  const mcy = 0.55 + ny * 0.3;
  const mh = 0.04 + 0.12 * p.mouthOpen;
  for (let i = 0; i < 12; i++) {
    const th = Math.PI + (2 * Math.PI * i) / 12;
    pts.push({ x: mcx + 0.25 * Math.cos(th), y: mcy + mh * Math.sin(th) * (i <= 6 ? 1 : 1) });
  }
  const ih = 0.01 + 0.1 * p.mouthOpen;
  for (let i = 0; i < 8; i++) {
    const th = Math.PI + (2 * Math.PI * i) / 8;
    pts.push({ x: mcx + 0.18 * Math.cos(th), y: mcy + ih * Math.sin(th) });
  }
  return pts;
}

export function makeFace(p: FaceParams, rnd: () => number): Point2D[] {
  const rad = (p.rollDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return template(p).map((q) => {
    const x = q.x * c - q.y * s;
    const y = q.x * s + q.y * c;
    return {
      x: p.cx + x * p.scale + (rnd() - 0.5) * 2 * p.noise,
      y: p.cy + y * p.scale + (rnd() - 0.5) * 2 * p.noise,
    };
  });
}

/** 21-landmark hand centred at (cx, cy) in normalized coords. */
export function makeHand(cx: number, cy: number, handedness: 'Left' | 'Right', rnd: () => number, pinch = 0.08): HandResult {
  const landmarks = [];
  for (let i = 0; i < 21; i++) {
    const ang = (i / 21) * Math.PI * 2;
    let x = cx + 0.04 * Math.cos(ang) + (rnd() - 0.5) * 0.004;
    let y = cy + 0.05 * Math.sin(ang) + (rnd() - 0.5) * 0.004;
    if (i === 0) { x = cx; y = cy + 0.06; } // wrist
    if (i === 9) { x = cx; y = cy - 0.01; } // middle MCP
    if (i === 4) { x = cx - pinch / 2; y = cy - 0.05; } // thumb tip
    if (i === 8) { x = cx + pinch / 2; y = cy - 0.05; } // index tip
    landmarks.push({ x, y, z: 0 });
  }
  return { landmarks, handedness };
}

export function makeObject(cls: ObjectClass, x: number, y: number, conf = 0.8, w = 0.15, h = 0.2): DetectedObject {
  return { class: cls, confidence: conf, bbox: { x, y, width: w, height: h } };
}
