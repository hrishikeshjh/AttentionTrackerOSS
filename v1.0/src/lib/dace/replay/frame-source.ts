/**
 * Frame sources for live and offline (replay) runs (FR8).
 *
 *   WebcamSource — wraps the dashboard's <video> element; t = wall clock.
 *   FileSource   — a LOCAL video file (File/Blob, via an object URL; nothing
 *                  is uploaded) stepped deterministically: frame k is the
 *                  media time k * stepMs, reached by seeking and awaiting
 *                  'seeked', so every preset replays exactly the same frames
 *                  with t = k * stepMs regardless of machine speed.
 */
import type { FrameInput } from '../measurement/provider';

export interface FrameSource {
  readonly width: number;
  readonly height: number;
  /** next frame, or null at the end */
  next(): Promise<FrameInput | null>;
  close(): void;
}

export class WebcamSource implements FrameSource {
  constructor(
    private readonly video: HTMLVideoElement,
    private readonly clock: () => number = () => Date.now()
  ) {}
  get width(): number {
    return this.video.videoWidth;
  }
  get height(): number {
    return this.video.videoHeight;
  }
  async next(): Promise<FrameInput | null> {
    if (this.video.readyState < 2 || !this.video.videoWidth) return null;
    return { image: this.video, width: this.video.videoWidth, height: this.video.videoHeight, t: this.clock() };
  }
  close(): void {
    /* the dashboard owns the stream */
  }
}

/** Media times (ms) visited by a FileSource: 0, step, 2·step, ... < duration. Pure; exported for tests. */
export function frameTimes(durationMs: number, stepMs: number): number[] {
  if (!(durationMs > 0) || !(stepMs > 0)) return [];
  const n = Math.floor((durationMs - 1e-6) / stepMs) + 1;
  return Array.from({ length: n }, (_, k) => k * stepMs);
}

export class FileSource implements FrameSource {
  private video: HTMLVideoElement;
  private url: string;
  private k = 0;
  private times: number[] = [];

  private constructor(file: Blob, private readonly stepMs: number) {
    this.video = document.createElement('video');
    this.video.muted = true;
    this.video.playsInline = true;
    this.video.preload = 'auto';
    this.url = URL.createObjectURL(file);
    this.video.src = this.url;
  }

  /** Open a local file; resolves once metadata (size, duration) is known. */
  static async open(file: Blob, stepMs = 200): Promise<FileSource> {
    const s = new FileSource(file, stepMs);
    await new Promise<void>((resolve, reject) => {
      s.video.onloadedmetadata = () => resolve();
      s.video.onerror = () => reject(new Error('could not decode the video file'));
    });
    s.times = frameTimes(s.video.duration * 1000, stepMs);
    return s;
  }

  get width(): number {
    return this.video.videoWidth;
  }
  get height(): number {
    return this.video.videoHeight;
  }
  get frameCount(): number {
    return this.times.length;
  }
  get position(): number {
    return this.k;
  }

  private seek(ms: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const done = () => {
        this.video.removeEventListener('seeked', done);
        resolve();
      };
      this.video.addEventListener('seeked', done);
      this.video.onerror = () => reject(new Error('seek failed'));
      this.video.currentTime = ms / 1000;
    });
  }

  async next(): Promise<FrameInput | null> {
    if (this.k >= this.times.length) return null;
    const t = this.times[this.k++];
    await this.seek(t);
    return { image: this.video, width: this.video.videoWidth, height: this.video.videoHeight, t };
  }

  close(): void {
    this.video.removeAttribute('src');
    this.video.load();
    URL.revokeObjectURL(this.url);
  }
}
