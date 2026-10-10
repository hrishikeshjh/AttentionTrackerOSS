/**
 * Head pose from a 4x4 facial transformation matrix (Annex A.1, FR1).
 *
 * Convention of the input (MediaPipe Face Landmarker
 * `facialTransformationMatrixes[i].data`, column-major): the matrix maps the
 * canonical face model into a right-handed camera space with x right, y UP,
 * z toward the viewer (OpenGL style; the camera looks along −z). In the
 * canonical model the face looks along +z, so an identity rotation is a
 * frontal face.
 *
 * Decomposition: R = R_y(yaw) · R_x(pitch) · R_z(roll). With y up, a positive
 * rotation about +x tilts the face normal (0,0,1) to (0, −sin φ, cos φ), i.e.
 * DOWN, so the extracted pitch is POSITIVE FOR HEAD DOWN (FR1):
 *   pitch = asin(−r12), yaw = atan2(r02, r22), roll = atan2(r10, r11).
 * The sign is confirmed on the scripted up/down pilot segments; `pitchSign`
 * (params) exists only to correct a measured inversion.
 */
import type { EulerDeg, Mat3, Mat4 } from './types';

const RAD = 180 / Math.PI;
const DEG = Math.PI / 180;

/** Extract the (scale-normalised) rotation part of a column-major 4x4 matrix as row-major 3x3. */
export function rotationFromMat4(m: Mat4): Mat3 | null {
  if (!m || m.length < 16) return null;
  const r: Mat3 = [
    [m[0], m[4], m[8]],
    [m[1], m[5], m[9]],
    [m[2], m[6], m[10]],
  ];
  // Normalise columns (the transform may carry a uniform scale).
  for (let j = 0; j < 3; j++) {
    const n = Math.hypot(r[0][j], r[1][j], r[2][j]);
    if (!Number.isFinite(n) || n === 0) return null;
    for (let i = 0; i < 3; i++) r[i][j] /= n;
  }
  return r;
}

const clamp1 = (v: number) => Math.max(-1, Math.min(1, v));

/** Euler angles (deg) from a y-up rotation matrix. +pitch = head down. */
export function eulerFromRotation(r: Mat3, pitchSign = 1): EulerDeg {
  const pitch = Math.asin(clamp1(-r[1][2])) * RAD;
  const yaw = Math.atan2(r[0][2], r[2][2]) * RAD;
  const roll = Math.atan2(r[1][0], r[1][1]) * RAD;
  return { pitch: pitchSign * pitch, yaw, roll };
}

/** Euler angles (deg) from a MediaPipe column-major 4x4 transform. */
export function eulerFromMatrix(m: Mat4, pitchSign = 1): EulerDeg | null {
  const r = rotationFromMat4(m);
  return r ? eulerFromRotation(r, pitchSign) : null;
}

function mul3(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
  return out;
}

/** Inverse of eulerFromRotation: R = R_y(yaw) · R_x(pitch) · R_z(roll), y-up frame. */
export function rotationFromEuler(e: EulerDeg): Mat3 {
  const p = e.pitch * DEG;
  const y = e.yaw * DEG;
  const r = e.roll * DEG;
  const rx: Mat3 = [
    [1, 0, 0],
    [0, Math.cos(p), -Math.sin(p)],
    [0, Math.sin(p), Math.cos(p)],
  ];
  const ry: Mat3 = [
    [Math.cos(y), 0, Math.sin(y)],
    [0, 1, 0],
    [-Math.sin(y), 0, Math.cos(y)],
  ];
  const rz: Mat3 = [
    [Math.cos(r), -Math.sin(r), 0],
    [Math.sin(r), Math.cos(r), 0],
    [0, 0, 1],
  ];
  return mul3(ry, mul3(rx, rz));
}

/** Column-major 4x4 from a rotation (+ optional translation/scale) — used by tests and replay fixtures. */
export function mat4FromRotation(r: Mat3, t: [number, number, number] = [0, 0, -50], scale = 1): number[] {
  return [
    r[0][0] * scale, r[1][0] * scale, r[2][0] * scale, 0,
    r[0][1] * scale, r[1][1] * scale, r[2][1] * scale, 0,
    r[0][2] * scale, r[1][2] * scale, r[2][2] * scale, 0,
    t[0], t[1], t[2], 1,
  ];
}

/** Δθ = θ − θ_ref (null if either is unknown). */
export function relativePitch(pitchDeg: number | null, refDeg: number | null): number | null {
  if (pitchDeg === null || refDeg === null) return null;
  return pitchDeg - refDeg;
}

/**
 * Convert a rotation expressed in the OpenCV camera frame (x right, y DOWN,
 * z forward) into the y-up frame used above: R_gl = F · R_cv · F, F = diag(1,−1,−1).
 */
export function cvToGlRotation(rcv: Mat3): Mat3 {
  const f = [1, -1, -1];
  return [
    [rcv[0][0] * f[0] * f[0], rcv[0][1] * f[0] * f[1], rcv[0][2] * f[0] * f[2]],
    [rcv[1][0] * f[1] * f[0], rcv[1][1] * f[1] * f[1], rcv[1][2] * f[1] * f[2]],
    [rcv[2][0] * f[2] * f[0], rcv[2][1] * f[2] * f[1], rcv[2][2] * f[2] * f[2]],
  ];
}
