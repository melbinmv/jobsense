import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "JobSense",
  description: "Real-time job market intelligence powered by RAG and Claude AI",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}