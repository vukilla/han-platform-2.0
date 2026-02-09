export function StepList({ steps }: { steps: { title: string; detail: string }[] }) {
  return (
    <div className="grid gap-4">
      {steps.map((step, index) => (
        <div key={step.title} className="flex items-start gap-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-black text-white text-sm font-semibold">
            {String(index + 1).padStart(2, "0")}
          </div>
          <div>
            <p className="text-lg font-semibold text-black">{step.title}</p>
            <p className="text-sm text-black/70">{step.detail}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
