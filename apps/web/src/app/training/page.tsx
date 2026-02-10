import { Suspense } from "react";
import TrainingClient from "./TrainingClient";

export default function TrainingPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-black/60">Loading…</div>}>
      <TrainingClient />
    </Suspense>
  );
}

