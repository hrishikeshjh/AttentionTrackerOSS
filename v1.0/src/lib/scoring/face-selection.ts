export type FaceBox = { x: number; y: number; width: number; height: number };
export type FaceCandidate<T> = { box: FaceBox; detection: T };

const DISAPPEAR_FRAMES_BEFORE_UNLOCK = 10;
const SWITCH_SCORE_MARGIN = 1.2; // new candidate must beat lock by 20%
const SWITCH_CONSECUTIVE_FRAMES = 5;

function scoreCandidate(box: FaceBox, frameWidth: number, frameHeight: number): number {
  const area = box.width * box.height;
  const areaNorm = Math.min(1, area / (frameWidth * frameHeight));

  const boxCenterX = box.x + box.width / 2;
  const boxCenterY = box.y + box.height / 2;
  const frameCenterX = frameWidth / 2;
  const frameCenterY = frameHeight / 2;
  const maxDist = Math.sqrt(frameCenterX ** 2 + frameCenterY ** 2);
  const dist = Math.sqrt((boxCenterX - frameCenterX) ** 2 + (boxCenterY - frameCenterY) ** 2);
  const centeredness = maxDist > 0 ? 1 - Math.min(1, dist / maxDist) : 1;

  return areaNorm * 0.7 + centeredness * 0.3;
}

function iou(a: FaceBox, b: FaceBox): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.width, b.x + b.width);
  const y2 = Math.min(a.y + a.height, b.y + b.height);
  const interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.width * a.height + b.width * b.height - interArea;
  return union > 0 ? interArea / union : 0;
}

/**
 * Locks onto a single "primary" face across frames when multiple faces are
 * detected, instead of blindly taking detections[0] (whose order is
 * detector-dependent, not an explicit selection policy, and can flip
 * frame-to-frame between two similarly-scored faces).
 *
 * Policy: prefer the largest, most-centered face, with hysteresis so a
 * transient false detection or a momentary size fluctuation doesn't cause
 * flicker — the current lock is kept unless it disappears for
 * DISAPPEAR_FRAMES_BEFORE_UNLOCK consecutive frames, or a new candidate
 * consistently outscores it by SWITCH_SCORE_MARGIN for
 * SWITCH_CONSECUTIVE_FRAMES consecutive frames.
 */
export class PrimaryFaceSelector {
  private lockedBox: FaceBox | null = null;
  private framesSinceSeen = 0;
  private switchCandidateStreak = 0;

  select<T>(candidates: FaceCandidate<T>[], frameWidth: number, frameHeight: number): FaceCandidate<T> | null {
    if (candidates.length === 0) {
      this.framesSinceSeen++;
      if (this.framesSinceSeen >= DISAPPEAR_FRAMES_BEFORE_UNLOCK) {
        this.lockedBox = null;
      }
      return null;
    }

    const scored = candidates.map((c) => ({ c, score: scoreCandidate(c.box, frameWidth, frameHeight) }));
    scored.sort((a, b) => b.score - a.score);
    const best = scored[0];

    if (this.lockedBox === null) {
      this.lockedBox = best.c.box;
      this.framesSinceSeen = 0;
      this.switchCandidateStreak = 0;
      return best.c;
    }

    // Try to find the candidate closest to the current lock (identity continuity).
    let lockMatch: { c: FaceCandidate<T>; score: number } | null = null;
    let lockMatchIoU = 0;
    for (const s of scored) {
      const overlap = iou(s.c.box, this.lockedBox);
      if (overlap > lockMatchIoU) {
        lockMatchIoU = overlap;
        lockMatch = s;
      }
    }

    if (lockMatch && lockMatchIoU > 0.1) {
      this.framesSinceSeen = 0;
      // Only switch away from the continuity match if some other candidate
      // has dominated for several consecutive frames.
      if (best.c !== lockMatch.c && best.score > lockMatch.score * SWITCH_SCORE_MARGIN) {
        this.switchCandidateStreak++;
        if (this.switchCandidateStreak >= SWITCH_CONSECUTIVE_FRAMES) {
          this.lockedBox = best.c.box;
          this.switchCandidateStreak = 0;
          return best.c;
        }
      } else {
        this.switchCandidateStreak = 0;
      }
      this.lockedBox = lockMatch.c.box;
      return lockMatch.c;
    }

    // Lock disappeared this frame (no overlapping candidate).
    this.framesSinceSeen++;
    if (this.framesSinceSeen >= DISAPPEAR_FRAMES_BEFORE_UNLOCK) {
      this.lockedBox = best.c.box;
      this.framesSinceSeen = 0;
      this.switchCandidateStreak = 0;
      return best.c;
    }

    return null;
  }

  reset(): void {
    this.lockedBox = null;
    this.framesSinceSeen = 0;
    this.switchCandidateStreak = 0;
  }
}
