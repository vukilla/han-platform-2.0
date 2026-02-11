"use client";

import type { ReactNode } from "react";
import { PrivyProvider } from "@privy-io/react-auth";

const PRIVY_TEST_APP_ID = "cmkgtr0dy0421ky0cqdrlltjl";

export function AppProviders({ children }: { children: ReactNode }) {
  const appId = process.env.NEXT_PUBLIC_PRIVY_APP_ID || PRIVY_TEST_APP_ID;

  return (
    <PrivyProvider
      appId={appId}
      config={{
        loginMethods: ["wallet", "email"],
        appearance: {
          theme: "light",
          accentColor: "#111111",
        },
      }}
    >
      {children}
    </PrivyProvider>
  );
}
