"use client";

import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { login } from "@/lib/api";
import { setToken } from "@/lib/auth";

export default function AuthPage() {
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setStatus("Signing in...");
    setError(null);
    try {
      const response = await login(email, name || undefined);
      setToken(response.token);
      setStatus("Signed in. You can start a demo now.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
      setStatus(null);
    }
  }

  return (
    <div className="mx-auto max-w-xl">
      <Card className="space-y-6">
        <div>
          <p className="section-eyebrow">Access</p>
          <h1 className="text-3xl font-semibold text-black">Sign in to HumanX Data Factory</h1>
          <p className="mt-2 text-sm">Email magic links now. Wallet connect follows in the incentives phase.</p>
        </div>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <label className="grid gap-2 text-sm font-semibold text-black">
            Email
            <input
              type="email"
              placeholder="you@lab.com"
              className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm text-black"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              required
            />
          </label>
          <label className="grid gap-2 text-sm font-semibold text-black">
            Name
            <input
              type="text"
              placeholder="Optional"
              className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm text-black"
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <Button type="submit" className="w-full">
            Send magic link
          </Button>
        </form>
        {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      </Card>
    </div>
  );
}
