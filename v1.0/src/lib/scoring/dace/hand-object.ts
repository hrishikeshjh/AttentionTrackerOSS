/**
 * Hand–object relation and hand-shape features (Annex A.2).
 *
 *   holding(h, b) = 1[IoU(box(h), b) ≥ 0.1]
 *   near(h, b)    = 1[‖centre(h) − centre(b)‖ < 0.5 · diag(b)]
 *   pinch_t       = 1[‖thumb_tip − index_tip‖ / ‖wrist − middle_MCP‖ < π_max]
 *
 * All inputs are image-normalized (x, y in [0, 1]). Distances are made
 * isotropic by scaling x by the frame aspect ratio (width / height), so a
 * 640x480 frame does not stretch horizontal distances. IoU is a ratio of
 * areas and is aspect-invariant.
 *
 * MediaPipe hand landmark indices: 0 wrist, 4 thumb tip, 8 index tip, 9 middle-finger MCP.
 */
import type { DaceParams } from './params';
import type { HandResult, NormalizedBBox } from './types';

export const WRIST = 0;
export const THUMB_TIP = 4;
export const INDEX_TIP = 8;
export const MIDDLE_MCP = 9;

type Pt = { x: number; y: number };

export function isoDistance(a: Pt, b: Pt, aspect: number): number {
  return Math.hypot((a.x - b.x) * aspect, a.y - b.y);
}

/** Axis-aligned box around all hand landmarks (centre format). */
export function handBox(landmarks: ReadonlyArray<Pt>): NormalizedBBox | null {
  if (!landmarks || landmarks.length === 0) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  return { x: (minX + maxX) / 2, y: (minY + maxY) / 2, width: maxX - minX, height: maxY - minY };
}

/** IoU of two centre-format boxes. */
export function iou(a: NormalizedBBox, b: NormalizedBBox): number {
  const ax1 = a.x - a.width / 2, ax2 = a.x + a.width / 2, ay1 = a.y - a.height / 2, ay2 = a.y + a.height / 2;
  const bx1 = b.x - b.width / 2, bx2 = b.x + b.width / 2, by1 = b.y - b.height / 2, by2 = b.y + b.height / 2;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(ax1, bx1));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(ay1, by1));
  const inter = iw * ih;
  const union = a.width * a.height + b.width * b.height - inter;
  return union > 0 ? inter / union : 0;
}

/** Diagonal of a box in isotropic units. */
export function boxDiag(b: NormalizedBBox, aspect: number): number {
  return Math.hypot(b.width * aspect, b.height);
}

export function holding(hand: HandResult, obj: NormalizedBBox, params: Pick<DaceParams, 'holdIoU'>): boolean {
  const hb = handBox(hand.landmarks);
  return hb !== null && iou(hb, obj) >= params.holdIoU;
}

export function near(hand: HandResult, obj: NormalizedBBox, params: Pick<DaceParams, 'nearDiagFactor' | 'frameAspect'>): boolean {
  const hb = handBox(hand.landmarks);
  if (!hb) return false;
  return isoDistance(hb, obj, params.frameAspect) < params.nearDiagFactor * boxDiag(obj, params.frameAspect);
}

/** ‖wrist − middle_MCP‖ (isotropic normalized units). */
export function handLength(hand: HandResult, aspect: number): number | null {
  const w = hand.landmarks[WRIST];
  const m = hand.landmarks[MIDDLE_MCP];
  if (!w || !m) return null;
  const d = isoDistance(w, m, aspect);
  return d > 0 ? d : null;
}

export function pinchRatio(hand: HandResult, aspect: number): number | null {
  const t = hand.landmarks[THUMB_TIP];
  const i = hand.landmarks[INDEX_TIP];
  const len = handLength(hand, aspect);
  if (!t || !i || len === null) return null;
  return isoDistance(t, i, aspect) / len;
}

/** pinch_t: ratio strictly below π_max. */
export function isPinch(hand: HandResult, params: Pick<DaceParams, 'piMax' | 'frameAspect'>): boolean {
  const r = pinchRatio(hand, params.frameAspect);
  return r !== null && r < params.piMax;
}

/**
 * Second person in frame (B.1): keep only hands whose wrist lies within
 * `handFaceMaxDistFaceWidths` face widths of the primary face box centre.
 * With no face box, hands are kept unchanged.
 */
export function assignHandsToFace(
  hands: readonly HandResult[],
  faceBox: NormalizedBBox | null,
  params: Pick<DaceParams, 'handFaceMaxDistFaceWidths' | 'frameAspect'>
): HandResult[] {
  if (!faceBox || !(faceBox.width > 0)) return [...hands];
  const maxD = params.handFaceMaxDistFaceWidths * faceBox.width * params.frameAspect;
  return hands.filter((h) => {
    const w = h.landmarks[WRIST];
    return !!w && isoDistance(w, faceBox, params.frameAspect) <= maxD;
  });
}
