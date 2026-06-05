import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Virtual Booth — CDJ-2000NXS / DJM-900NXS Trainer',
  description: 'Real Web Audio DJ trainer. Two CDJ-2000NXS decks + DJM-900NXS mixer in the real spatial layout. Learn to beatmatch, cue, EQ swap, and mix.',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
