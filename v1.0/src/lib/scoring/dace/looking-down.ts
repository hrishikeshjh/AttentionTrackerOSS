/**
 * Looking down (Annex A.1):  D_t = 1[Δθ_t ≥ θ_down  or  g_t ≥ γ_down]   (both inclusive)
 *
 * The gaze clause is used only when gaze is valid (iris present and
 * consistent); otherwise D_t falls back to head pitch only (B.1). With
 * neither signal available, D_t = 0.
 */
import type { DaceParams } from './params';

export function isLookingDown(
  relPitchDeg: number | null,
  gazeV: number | null,
  gazeValid: boolean,
  params: Pick<DaceParams, 'thetaDownDeg' | 'gammaDown'>
): boolean {
  const head = relPitchDeg !== null && relPitchDeg >= params.thetaDownDeg;
  const eyes = gazeValid && gazeV !== null && gazeV >= params.gammaDown;
  return head || eyes;
}
