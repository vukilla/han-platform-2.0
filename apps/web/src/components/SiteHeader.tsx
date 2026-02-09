import Link from "next/link";
import { LinkButton } from "./ui/Button";

const nav = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/demos/new", label: "New Demo" },
  { href: "/training", label: "Training" },
  { href: "/rewards", label: "Rewards" },
];

export function SiteHeader() {
  return (
    <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-6 py-6">
      <Link href="/" className="text-lg font-semibold tracking-tight">
        HumanX Data Factory
      </Link>
      <nav className="hidden items-center gap-6 text-sm font-semibold text-black/70 md:flex">
        {nav.map((item) => (
          <Link key={item.href} href={item.href} className="transition hover:text-black">
            {item.label}
          </Link>
        ))}
      </nav>
      <LinkButton href="/auth" variant="outline">
        Sign in
      </LinkButton>
    </header>
  );
}
