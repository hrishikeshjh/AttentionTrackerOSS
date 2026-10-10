/**
 * Desk-zone evidence (Annex A.2, FR5): per-frame features and window aggregation.
 *
 *   P_t = 1[phone detected ∧ holding(h, phone) ∧ D_t]
 *   phone in use   : ΣP_t over W ≥ ρ_use · |W|
 *   phone present  : phone detected in ≥ 30% of W but not in use (never lowers the score)
 *   book           : book detected and held or near a hand in ≥ 50% of W
 *   writing        : pinch_t and move_t each in ≥ 60% of W (same hand), D_t in most
 *                    of W (> 50%), and no phone held in W. Both hands are checked
 *                    independently (left-handed writers, B.1).
 *   desk visible V : in ≥ 30% of the last 25 frames, a hand wrist is below the
 *                    bottom edge of the face box, or a book/phone is detected.
 *   conflict (U4)  : phone in use and pen-grip motion (pinch ∧ move shares) in a
 *                    hand other than the phone hand (plan decision D11).
 *
 * |W| is always params.windowFrames (15): frames not yet observed at session
 * start count as "condition not met", which gives a natural warm-up.
 * Detections below c_min are ignored. 'book' and 'notebook' count as book.
 */
import type { DaceParams } from './params';
import { atLeastShare, moreThanShare } from './params';
import type {
  DaceFrame,
  DetectedObject,
  FrameEvidence,
  HandFrameFeatures,
  HandId,
  HandResult,
  WindowSummary,
} from './types';
import { WRIST, handLength, holding, isoDistance, isPinch, near, pinchRatio } from './hand-object';

export const isPhone = (o: DetectedObject) => o.class === 'phone';
export const isBook = (o: DetectedObject) => o.class === 'book' || o.class === 'notebook';

export function filterByConfidence(objects: readonly DetectedObject[], cMin: number): DetectedObject[] {
  return objects.filter((o) => o.confidence >= cMin);
}

/** Assign stable ids by handedness; a duplicate label takes the free one; extra hands are dropped. */
export function assignHandIds(hands: readonly HandResult[]): Array<{ id: HandId; hand: HandResult }> {
  const out: Array<{ id: HandId; hand: HandResult }> = [];
  const used = new Set<HandId>();
  for (const h of hands) {
    let id: HandId = h.handedness === 'Right' ? 'Right' : 'Left';
    if (used.has(id)) id = id === 'Left' ? 'Right' : 'Left';
    if (used.has(id)) continue;
    used.add(id);
    out.push({ id, hand: h });
  }
  return out;
}

export function computeFrameEvidence(
  frame: DaceFrame,
  lookingDown: boolean,
  prev: FrameEvidence | null,
  params: DaceParams
): FrameEvidence {
  const aspect = params.frameAspect;
  const objects = filterByConfidence(frame.objects, params.cMin);
  const phones = objects.filter(isPhone);
  const books = objects.filter(isBook);

  const hands: HandFrameFeatures[] = [];
  for (const { id, hand } of assignHandIds(frame.hands)) {
    const len = handLength(hand, aspect);
    const wrist = hand.landmarks[WRIST];
    if (len === null || !wrist) continue;
    hands.push({
      id,
      pinch: isPinch(hand, params),
      pinchRatio: pinchRatio(hand, aspect),
      wrist: { x: wrist.x, y: wrist.y },
      handLength: len,
      holdingPhone: phones.some((p) => holding(hand, p.bbox, params)),
      holdingBook: books.some((b) => holding(hand, b.bbox, params)),
      nearBook: books.some((b) => near(hand, b.bbox, params)),
    });
  }

  const move: Partial<Record<HandId, boolean>> = {};
  const moveAmount: Partial<Record<HandId, number | null>> = {};
  for (const h of hands) {
    const p = prev?.hands.find((q) => q.id === h.id);
    if (!p) {
      move[h.id] = false;
      moveAmount[h.id] = null;
      continue;
    }
    const d = isoDistance(h.wrist, p.wrist, aspect) / h.handLength;
    moveAmount[h.id] = d;
    move[h.id] = d >= params.mMin && d <= params.mMax;
  }

  const face = frame.face;
  const faceBox = face?.faceBox ?? null;
  const phoneDetected = phones.length > 0;
  const bookDetected = books.length > 0;
  const wristBelowFace =
    faceBox !== null && hands.some((h) => h.wrist.y > faceBox.y + faceBox.height / 2);

  return {
    t: frame.t,
    faceVisible: face !== null,
    lookingDown: face !== null && lookingDown,
    eyeClosed: face?.eyeClosed ?? null,
    phoneDetected,
    bookDetected,
    phoneInUseFrame: phoneDetected && hands.some((h) => h.holdingPhone) && face !== null && lookingDown,
    bookEvidenceFrame: bookDetected && hands.some((h) => h.holdingBook || h.nearBook),
    deskCue: wristBelowFace || phoneDetected || bookDetected,
    hands,
    move,
    moveAmount,
    relPitchDeg: face?.relPitchDeg ?? null,
    yawDeg: face?.yawDeg ?? null,
    gazeH: face?.gazeValid ? (face?.gazeH ?? null) : null,
  };
}

export type EvidenceSummary = Omit<WindowSummary, 't' | 'downEpisodeMs' | 'inDownEpisode'>;

/**
 * Aggregate the last |W| frames (`win`) and the last 25 frames (`desk`).
 * Both arrays are oldest -> newest.
 */
export function summarizeEvidence(
  win: readonly FrameEvidence[],
  desk: readonly FrameEvidence[],
  params: DaceParams
): EvidenceSummary {
  const N = params.windowFrames;
  const count = (f: (e: FrameEvidence) => boolean) => win.reduce((a, e) => a + (f(e) ? 1 : 0), 0);

  const noFaceCount = count((e) => !e.faceVisible);
  const lookingDownCount = count((e) => e.lookingDown);
  const lookingDownMost = moreThanShare(lookingDownCount, N, params.mostShare);
  const phoneInUseCount = count((e) => e.phoneInUseFrame);
  const phoneDetectedCount = count((e) => e.phoneDetected);
  const phoneInUse = atLeastShare(phoneInUseCount, N, params.rhoUse);
  const phonePresent = !phoneInUse && atLeastShare(phoneDetectedCount, N, params.phonePresentShare);
  const phoneHoldCount: Partial<Record<HandId, number>> = {};
  for (const e of win) for (const h of e.hands) if (h.holdingPhone) phoneHoldCount[h.id] = (phoneHoldCount[h.id] ?? 0) + 1;
  const phoneHeldAny = Object.values(phoneHoldCount).some((c) => (c ?? 0) > 0);
  const bookEvidenceCount = count((e) => e.bookEvidenceFrame);
  const book = atLeastShare(bookEvidenceCount, N, params.bookShare);

  const penGripMotion: Partial<Record<HandId, boolean>> = {};
  for (const id of ['Left', 'Right'] as HandId[]) {
    const pinchC = count((e) => e.hands.some((h) => h.id === id && h.pinch));
    const moveC = count((e) => e.move[id] === true);
    if (pinchC > 0 || moveC > 0) {
      penGripMotion[id] = atLeastShare(pinchC, N, params.writingShare) && atLeastShare(moveC, N, params.writingShare);
    }
  }
  const anyPenGrip = Object.values(penGripMotion).some(Boolean);
  const writing = anyPenGrip && lookingDownMost && !phoneHeldAny;

  let conflict = false;
  if (phoneInUse) {
    const maxHold = Math.max(0, ...Object.values(phoneHoldCount).map((c) => c ?? 0));
    const phoneHands = (Object.keys(phoneHoldCount) as HandId[]).filter((id) => (phoneHoldCount[id] ?? 0) === maxHold && maxHold > 0);
    conflict = (Object.keys(penGripMotion) as HandId[]).some((id) => penGripMotion[id] && !phoneHands.includes(id));
  }

  const deskN = params.deskWindowFrames;
  const deskCueCount = desk.reduce((a, e) => a + (e.deskCue ? 1 : 0), 0);
  const deskVisible = atLeastShare(deskCueCount, deskN, params.deskShare);

  // Rule 7 inputs: among face frames not looking down, majority meet the screen criteria.
  const eligible = win.filter((e) => e.faceVisible && !e.lookingDown);
  const onScreen = eligible.filter(
    (e) =>
      (e.gazeH === null || Math.abs(e.gazeH) <= params.screenGazeMax) &&
      (e.yawDeg === null || Math.abs(e.yawDeg) <= params.screenYawMaxDeg) &&
      (e.relPitchDeg === null || e.relPitchDeg >= -params.screenPitchUpMaxDeg)
  ).length;
  const screenAttentive = eligible.length > 0 && onScreen * 2 > eligible.length;

  return {
    n: N,
    noFaceCount,
    lookingDownCount,
    lookingDownMost,
    phoneInUseCount,
    phoneDetectedCount,
    phoneInUse,
    phonePresent,
    phoneHeldAny,
    bookEvidenceCount,
    book,
    penGripMotion,
    writing,
    conflict,
    deskCueCount,
    deskWindowN: deskN,
    deskVisible,
    screenAttentive,
  };
}
