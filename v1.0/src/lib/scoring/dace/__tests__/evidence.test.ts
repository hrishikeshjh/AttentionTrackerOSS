import { describe, it, expect } from 'vitest';
import { iou, holding, near, isPinch, pinchRatio, handBox, assignHandsToFace } from '../hand-object';
import { DaceWindow } from '../window';
import { assignHandIds, filterByConfidence } from '../evidence';
import { makeParams } from '../params';
import { P, down, face, frame, hand, obj, writingHand, FACE_BOX } from './helpers';
import type { DaceFrame } from '../types';

const ISO = makeParams({ frameAspect: 1 });

function run(frames: DaceFrame[], params = P) {
  const w = new DaceWindow(params);
  frames.forEach((f) => w.push(f));
  return w.summary();
}

/** 15 frames where phone-in-use evidence holds in the first `k` (others: phone visible, not held). */
function phoneFrames(k: number) {
  return Array.from({ length: 15 }, (_, i) => {
    const held = i < k;
    return frame(i, down(), held ? [hand('Right', 0.5, 0.8)] : [hand('Right', 0.1, 0.8)], [obj('phone', 0.5, 0.81)]);
  });
}

describe('hand–object relation', () => {
  it('IoU ≥ 0.1 is holding (inclusive), below is not', () => {
    const a = { x: 0.5, y: 0.5, width: 0.2, height: 0.2 };
    // shift b horizontally so that IoU = 0.1 exactly: inter = 0.2*w, union = 0.08 - 0.2w -> w = 0.04/1.2... use iou() to search
    const b = (dx: number) => ({ x: 0.5 + dx, y: 0.5, width: 0.2, height: 0.2 });
    // IoU(dx) = (0.2-dx)/(0.2+dx) -> 0.1 at dx = 0.2*0.9/1.1
    const dx = (0.2 * 0.9) / 1.1;
    expect(iou(a, b(dx))).toBeCloseTo(0.1, 12);
    const handAt = (cx: number) => ({ landmarks: [{ x: cx - 0.1, y: 0.4, z: 0 }, { x: cx + 0.1, y: 0.6, z: 0 }], handedness: 'Left' as const });
    expect(holding(handAt(0.5), b(dx - 1e-6), P)).toBe(true);
    expect(holding(handAt(0.5), b(dx + 1e-3), P)).toBe(false);
  });

  it('near: ‖centre(h) − centre(b)‖ < 0.5 · diag(b) (strict, isotropic)', () => {
    const b = { x: 0.5, y: 0.5, width: 0.06, height: 0.08 }; // diag 0.1 -> radius 0.05
    const handAtY = (cy: number) => ({ landmarks: [{ x: 0.49, y: cy - 0.01, z: 0 }, { x: 0.51, y: cy + 0.01, z: 0 }], handedness: 'Left' as const });
    expect(near(handAtY(0.5 + 0.0499), b, ISO)).toBe(true);
    expect(near(handAtY(0.5 + 0.0501), b, ISO)).toBe(false);
  });

  it('pinch ratio < π_max (strict); both hands treated the same', () => {
    expect(pinchRatio(hand('Left', 0.5, 0.5, { tipGap: 0.021 }), 1)).toBeCloseTo(0.35, 12);
    expect(isPinch(hand('Left', 0.5, 0.5, { tipGap: 0.021 }), ISO)).toBe(false);
    expect(isPinch(hand('Left', 0.5, 0.5, { tipGap: 0.0209 }), ISO)).toBe(true);
    expect(isPinch(hand('Right', 0.5, 0.5, { tipGap: 0.0209 }), ISO)).toBe(true);
  });

  it('hand box, confidence filter, ids and second-person filtering', () => {
    expect(handBox([])).toBeNull();
    expect(filterByConfidence([obj('phone', 0.5, 0.5, 0.1, 0.1, 0.35), obj('book', 0.5, 0.5, 0.1, 0.1, 0.3499)], 0.35)).toHaveLength(1);
    const ids = assignHandIds([hand('Left', 0.2, 0.8), hand('Left', 0.7, 0.8), hand('Left', 0.9, 0.8)]);
    expect(ids.map((h) => h.id)).toEqual(['Left', 'Right']);
    const far = hand('Left', 0.05, 0.1);
    far.landmarks[0] = { x: 1.6, y: 0.1, z: 0 };
    expect(assignHandsToFace([hand('Right', 0.5, 0.8), far], FACE_BOX, P)).toHaveLength(1);
  });
});

describe('phone in use / present (ρ_use = 0.6 of |W| = 15)', () => {
  it('9/15 frames of P_t -> in use; 8/15 -> not in use but present', () => {
    expect(run(phoneFrames(9)).phoneInUse).toBe(true);
    const s8 = run(phoneFrames(8));
    expect(s8.phoneInUse).toBe(false);
    expect(s8.phonePresent).toBe(true);
  });

  it('P_t needs D_t: holding a phone while looking at the screen is not "in use"', () => {
    const fs = Array.from({ length: 15 }, (_, i) => frame(i, face(), [hand('Right', 0.5, 0.8)], [obj('phone', 0.5, 0.81)]));
    const s = run(fs);
    expect(s.phoneInUse).toBe(false);
    expect(s.phonePresent).toBe(true);
  });

  it('phone present needs ≥ 30% of W (5/15 yes, 4/15 no); detections below c_min ignored', () => {
    const mk = (k: number, conf = 0.9) =>
      Array.from({ length: 15 }, (_, i) => frame(i, down(), [], i < k ? [obj('phone', 0.5, 0.81, 0.06, 0.1, conf)] : []));
    expect(run(mk(5)).phonePresent).toBe(true);
    expect(run(mk(4)).phonePresent).toBe(false);
    expect(run(mk(15, 0.34)).phoneDetectedCount).toBe(0);
  });
});

describe('book evidence (≥ 50% of W held or near)', () => {
  const mk = (k: number) =>
    Array.from({ length: 15 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.8)], i < k ? [obj('book', 0.5, 0.82, 0.2, 0.15)] : []));
  it('8/15 -> book; 7/15 -> no book', () => {
    expect(run(mk(8)).book).toBe(true);
    expect(run(mk(7)).book).toBe(false);
  });
  it('a book far from any hand is not evidence', () => {
    const fs = Array.from({ length: 15 }, (_, i) => frame(i, down(), [hand('Right', 0.1, 0.8)], [obj('book', 0.8, 0.8, 0.1, 0.1)]));
    expect(run(fs).book).toBe(false);
  });
  it('notebook counts as book', () => {
    const fs = Array.from({ length: 15 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.8)], [obj('notebook', 0.5, 0.82, 0.2, 0.15)]));
    expect(run(fs).book).toBe(true);
  });
});

describe('writing detection (all clauses)', () => {
  const writing = (o: { k?: number; handed?: 'Left' | 'Right'; downFrames?: number; pinch?: boolean; still?: boolean; phoneHeld?: boolean } = {}) =>
    Array.from({ length: 15 }, (_, i) => {
      const handed = o.handed ?? 'Right';
      const active = i < (o.k ?? 15);
      // inactive frames: hand jumps away (displacement > m_max) and stays still, no pinch
      let h = active ? writingHand(handed, i) : hand(handed, 0.6, 0.9);
      if (o.pinch === false) h = hand(handed, 0.6, 0.8 + (i % 2 ? 0.0015 : -0.0015));
      if (o.still) h = hand(handed, 0.6, 0.8, { pinch: true });
      const hands = [h];
      const objects = [];
      if (o.phoneHeld && i === 3) {
        hands.push(hand(handed === 'Right' ? 'Left' : 'Right', 0.2, 0.8));
        objects.push(obj('phone', 0.2, 0.81));
      }
      return frame(i, i < (o.downFrames ?? 15) ? down() : face(), hands, objects);
    });

  it('pinch + motion + looking down in most of W -> writing (right hand)', () => {
    const s = run(writing());
    expect(s.penGripMotion.Right).toBe(true);
    expect(s.writing).toBe(true);
  });
  it('left-handed writer is detected the same way', () => {
    const s = run(writing({ handed: 'Left' }));
    expect(s.penGripMotion.Left).toBe(true);
    expect(s.writing).toBe(true);
  });
  it('pinch and move must each hold in ≥ 60% of W: 9/15 yes, 8/15 no', () => {
    // move needs the previous frame of the same hand; frame 0 has no move, so k=10 gives 9 move frames.
    expect(run(writing({ k: 10 })).writing).toBe(true);
    expect(run(writing({ k: 9 })).writing).toBe(false);
  });
  it('no pinch -> no writing; no wrist motion -> no writing', () => {
    expect(run(writing({ pinch: false })).writing).toBe(false);
    expect(run(writing({ still: true })).writing).toBe(false);
  });
  it('wrist motion above m_max is not writing', () => {
    const fs = Array.from({ length: 15 }, (_, i) => frame(i, down(), [hand('Right', 0.6, 0.8 + (i % 2 ? 0.006 : -0.006), { pinch: true })]));
    // 0.012 / 0.06 = 0.2 hand lengths > 0.15
    expect(run(fs).writing).toBe(false);
  });
  it('D_t must hold in most of W (8/15 yes, 7/15 no)', () => {
    expect(run(writing({ downFrames: 8 })).writing).toBe(true);
    expect(run(writing({ downFrames: 7 })).writing).toBe(false);
  });
  it('a phone held anywhere in W disables writing', () => {
    const s = run(writing({ phoneHeld: true }));
    expect(s.phoneHeldAny).toBe(true);
    expect(s.writing).toBe(false);
  });
});

describe('U4 conflict: phone in use in one hand + pen-grip motion in the other', () => {
  it('fires only when the pen-grip hand is not the phone hand', () => {
    const fs = Array.from({ length: 15 }, (_, i) =>
      frame(i, down(), [hand('Left', 0.3, 0.8), writingHand('Right', i, 0.7)], [obj('phone', 0.3, 0.81)])
    );
    const s = run(fs);
    expect(s.phoneInUse).toBe(true);
    expect(s.conflict).toBe(true);
    expect(s.writing).toBe(false); // phone held
    const one = Array.from({ length: 15 }, (_, i) => frame(i, down(), [hand('Left', 0.3, 0.8)], [obj('phone', 0.3, 0.81)]));
    expect(run(one).conflict).toBe(false);
  });
});

describe('desk visibility V (≥ 30% of the last 25 frames)', () => {
  const mk = (k: number) =>
    Array.from({ length: 25 }, (_, i) => frame(i, down(), i < k ? [hand('Right', 0.5, 0.8)] : []));
  it('8/25 -> visible; 7/25 -> not visible', () => {
    expect(run(mk(8)).deskVisible).toBe(true);
    expect(run(mk(7)).deskVisible).toBe(false);
  });
  it('a hand above the face-box bottom is not a desk cue; a book/phone always is', () => {
    const high = Array.from({ length: 25 }, (_, i) => frame(i, down(), [hand('Right', 0.5, 0.2)]));
    expect(run(high).deskVisible).toBe(false);
    const objs = Array.from({ length: 25 }, (_, i) => frame(i, null, [], [obj('book', 0.5, 0.9)]));
    expect(run(objs).deskVisible).toBe(true);
  });
});

describe('downward episode tracking', () => {
  it('15 down frames = 3000 ms; one-frame gap tolerated; two-frame gap ends the episode', () => {
    const w = new DaceWindow(P);
    for (let i = 0; i < 15; i++) w.push(frame(i, down()));
    expect(w.summary().downEpisodeMs).toBe(3000);
    w.push(frame(15, face()));
    expect(w.summary().inDownEpisode).toBe(true);
    w.push(frame(16, down()));
    expect(w.summary().downEpisodeMs).toBe(3400);
    w.push(frame(17, face()));
    w.push(frame(18, face()));
    expect(w.summary().inDownEpisode).toBe(false);
    expect(w.summary().downEpisodeMs).toBe(0);
  });
  it('no-face frames count toward absence, not looking down', () => {
    const s = run(Array.from({ length: 15 }, (_, i) => frame(i, i < 12 ? null : face())));
    expect(s.noFaceCount).toBe(12);
    expect(s.lookingDownCount).toBe(0);
  });
});
