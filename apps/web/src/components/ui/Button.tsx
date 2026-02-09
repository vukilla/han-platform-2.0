import Link from "next/link";
import { ComponentProps } from "react";

const base =
  "inline-flex items-center justify-center gap-2 rounded-full px-5 py-2 text-sm font-semibold transition";

const variants = {
  primary:
    "bg-black text-white shadow-[0_12px_30px_rgba(0,0,0,0.25)] hover:-translate-y-0.5 hover:shadow-[0_18px_40px_rgba(0,0,0,0.3)]",
  outline:
    "border border-black/20 text-black hover:border-black hover:bg-black hover:text-white",
  ghost: "text-black/70 hover:text-black",
};

type ButtonProps = ComponentProps<"button"> & {
  variant?: keyof typeof variants;
};

type LinkButtonProps = ComponentProps<typeof Link> & {
  variant?: keyof typeof variants;
};

export function Button({ variant = "primary", className = "", ...props }: ButtonProps) {
  return <button className={`${base} ${variants[variant]} ${className}`} {...props} />;
}

export function LinkButton({ variant = "primary", className = "", ...props }: LinkButtonProps) {
  return <Link className={`${base} ${variants[variant]} ${className}`} {...props} />;
}
