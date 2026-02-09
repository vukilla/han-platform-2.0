export function Badge({ label, tone = "stone" }: { label: string; tone?: "stone" | "amber" | "emerald" | "rose" }) {
  const tones = {
    stone: "bg-stone-100 text-stone-700",
    amber: "bg-amber-100 text-amber-700",
    emerald: "bg-emerald-100 text-emerald-700",
    rose: "bg-rose-100 text-rose-700",
  };

  return (
    <span className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ${tones[tone]}`}>
      {label}
    </span>
  );
}
