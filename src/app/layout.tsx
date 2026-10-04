/**
 * Root layout for the App Router.
 *
 * Server Component, copied from transformer-explainer's layout: HTML
 * scaffold, the global stylesheet and the site header. Dark mode follows
 * the system setting (`darkMode: "media"` in tailwind.config.ts).
 */
import type { Metadata } from "next";
import type { ReactNode } from "react";

import { SiteHeader } from "@/components/ui/SiteHeader";
import { SITE_URL } from "@/lib/site";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "LLM Inference Explained",
    template: "%s · LLM Inference Explained",
  },
  description:
    "An interactive explainer of real LLM inference: the generation loop, the KV cache, rooflines, batching, paging, attention kernels, speculative decoding, quantisation, parallelism and serving metrics.",
};

export default function RootLayout({
  children,
}: {
  children: ReactNode;
}): JSX.Element {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans antialiased">
        <SiteHeader />
        {children}
      </body>
    </html>
  );
}
