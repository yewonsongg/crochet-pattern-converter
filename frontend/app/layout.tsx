import type { Metadata } from "next";
import { Gaegu } from "next/font/google";
import "./globals.css";

const gaegu = Gaegu({
  variable: "--font-gaegu",
  weight: ["400", "700"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "StitchScript — Crochet Pattern Converter",
  description: "Turn crochet chart PDFs into clear, editable written patterns.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={gaegu.variable}>
      <body>{children}</body>
    </html>
  );
}
