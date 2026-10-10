/**
 * DACE pure core (Stage 1). Deliberately NOT re-exported from scoring/index.ts,
 * so nothing on the legacy path imports DACE code.
 */
export * from './VERSION';
export * from './types';
export * from './params';
export * from './head-pose-matrix';
export * from './pitch-reference';
export * from './gaze-blendshape';
export * from './eye-openness';
export * from './perclos';
export * from './looking-down';
export * from './solvepnp';
export * from './mesh-adapter';
export * from './hand-object';
export * from './evidence';
export * from './window';
export * from './rules';
export * from './abstain';
export * from './conformal';
export * from './kinematics';
export * from './feature-mapping';
export * from './feature-vector';
export * from './score-hold';
