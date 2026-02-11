import type { Metadata } from "next";
import { Space_Grotesk, Work_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { SiteHeader } from "@/components/SiteHeader";
import { AppProviders } from "@/components/AppProviders";

const display = Space_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
});

const body = Work_Sans({
  variable: "--font-body",
  subsets: ["latin"],
});

const mono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Humanoid Network",
  description: "Upload videos, generate interaction datasets, and train policies.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${display.variable} ${body.variable} ${mono.variable} antialiased`}>
        <AppProviders>
          <div className="relative min-h-screen overflow-hidden">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(255,200,120,0.35),_transparent_55%),radial-gradient(circle_at_30%_30%,_rgba(102,146,255,0.25),_transparent_50%),radial-gradient(circle_at_80%_80%,_rgba(255,124,91,0.18),_transparent_45%)]" />
            <div className="relative">
              <SiteHeader />
              <main className="mx-auto w-full max-w-6xl px-6 pb-16">{children}</main>
            </div>
          </div>
        </AppProviders>
      </body>
    </html>
  );
}
