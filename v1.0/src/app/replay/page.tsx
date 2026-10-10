import { notFound } from 'next/navigation';
import ReplayClient from './replay-client';

/**
 * Research-only offline replay (FR8): run a LOCAL video file through the
 * Stage 1 ablation presets and download one JSONL eval log per preset.
 * Returns 404 unless NEXT_PUBLIC_DACE_RESEARCH=1. Nothing is uploaded.
 */
export default function ReplayPage() {
  if (process.env.NEXT_PUBLIC_DACE_RESEARCH !== '1') notFound();
  return <ReplayClient />;
}
