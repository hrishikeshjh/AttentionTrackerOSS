export type AttentionState = 'Attentive' | 'Distracted' | 'Multitasking' | 'Idle';

export type SessionDataPoint = {
  timestamp: number;
  attentionScore: number;
  attentionState: AttentionState;
};

export type SessionSummary = {
  averageAttention: number;
  multitaskEvents: number;
  engagementLevel: 'High' | 'Moderate' | 'Low';
  duration: string;
};

export type FaceMetrics = {
  ear: number | null;
  mar: number | null;
  yaw: number | null;
  pitch: number | null;
  roll: number | null;
};
