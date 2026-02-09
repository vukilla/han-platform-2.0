import { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={`rounded-3xl border border-black/10 bg-white/80 p-6 shadow-[0_18px_40px_rgba(17,24,39,0.08)] backdrop-blur ${className}`}
    >
      {children}
    </div>
  );
}
