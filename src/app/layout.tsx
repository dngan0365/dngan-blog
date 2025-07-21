import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Header from "@/components/Header";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "My Blog",
    template: "%s | My Blog",
  },
  description: "A personal blog about web development, technology, and more.",
  keywords: ["blog", "web development", "technology", "programming"],
  authors: [{ name: "Blog Author" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://myblog.com",
    siteName: "My Blog",
    title: "My Blog",
    description: "A personal blog about web development, technology, and more.",
  },
  twitter: {
    card: "summary_large_image",
    title: "My Blog",
    description: "A personal blog about web development, technology, and more.",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Header />
        <main>{children}</main>
      </body>
    </html>
  );
}
