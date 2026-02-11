"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { usePrivy } from "@privy-io/react-auth";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { login } from "@/lib/api";
import { clearToken, getToken, setToken } from "@/lib/auth";

type PrivyAccount = {
  type?: string;
  address?: string;
};

type PrivyUserLike = {
  id?: string;
  email?: { address?: string };
  wallet?: { address?: string };
  linkedAccounts?: PrivyAccount[];
};

function resolveIdentity(user: PrivyUserLike): { email: string; name?: string } {
  const email = user?.email?.address?.trim();
  if (email) {
    return { email };
  }

  const walletAddress =
    user?.wallet?.address ??
    user?.linkedAccounts?.find((account) => account?.type === "wallet")?.address;
  if (walletAddress) {
    const lower = walletAddress.toLowerCase();
    return {
      email: `${lower}@privy.local`,
      name: `Wallet ${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}`,
    };
  }

  const fallbackId = user?.id ?? "user";
  return { email: `privy-${fallbackId}@privy.local`, name: "Privy User" };
}

export default function AuthPage() {
  const { ready, authenticated, user, login: privyLogin, logout: privyLogout } = usePrivy();
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState(false);

  const identity = useMemo(() => resolveIdentity((user ?? {}) as PrivyUserLike), [user]);

  const syncApiSession = useCallback(async () => {
    if (!authenticated || !ready || !user) {
      return;
    }
    setSyncing(true);
    setError(null);
    setStatus("Signing in...");
    try {
      const response = await login(identity.email, identity.name);
      setToken(response.token);
      setStatus("Signed in. You can start a demo now.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
      setStatus(null);
    } finally {
      setSyncing(false);
    }
  }, [authenticated, ready, user, identity.email, identity.name]);

  useEffect(() => {
    if (!ready || !authenticated || !user) {
      return;
    }
    if (getToken()) {
      setStatus("Signed in. You can start a demo now.");
      return;
    }
    void syncApiSession();
  }, [ready, authenticated, user, syncApiSession]);

  async function handlePrivySignIn() {
    setError(null);
    setStatus("Opening Privy...");
    await privyLogin();
  }

  async function handleSignOut() {
    clearToken();
    await privyLogout();
    setStatus("Signed out.");
  }

  return (
    <div className="mx-auto max-w-xl">
      <Card className="space-y-6">
        <div>
          <p className="section-eyebrow">Access</p>
          <h1 className="text-3xl font-semibold text-black">Sign in to Humanoid Network</h1>
          <p className="mt-2 text-sm">Use Privy to sign in with a wallet or email.</p>
        </div>
        <div className="space-y-4">
          <Button onClick={handlePrivySignIn} className="w-full" disabled={!ready || authenticated || syncing}>
            {ready ? "Sign in with Privy" : "Loading Privy..."}
          </Button>
          {authenticated ? (
            <>
              <div className="rounded-2xl border border-black/10 bg-black/[0.02] px-4 py-3 text-sm text-black/70">
                <p>Authenticated as: {identity.email}</p>
                {identity.name ? <p>Display name: {identity.name}</p> : null}
              </div>
              <Button onClick={syncApiSession} className="w-full" disabled={syncing}>
                {syncing ? "Syncing session..." : "Sync app session"}
              </Button>
              <Button onClick={handleSignOut} className="w-full" variant="outline">
                Sign out
              </Button>
            </>
          ) : null}
        </div>
        {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      </Card>
    </div>
  );
}
