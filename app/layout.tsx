import type { Metadata, Viewport } from 'next';
import './globals.css';

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
};

export const metadata: Metadata = {
  title: 'Virtual Booth — CDJ-2000NXS / DJM-900NXS Trainer',
  description: 'Real Web Audio DJ trainer. Two CDJ-2000NXS decks + DJM-900NXS mixer. Works on desktop, iPad, and iPhone.',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Virtual Booth',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
