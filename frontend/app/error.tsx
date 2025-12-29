'use client';

import { useEffect } from 'react';
import Link from 'next/link';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Global error boundary:', error);
  }, [error]);

  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-950 text-slate-50">
        <div className="ui-container py-16">
          <div className="ui-surface p-8">
            <h1 className="text-2xl font-semibold">Something went wrong</h1>
            <p className="mt-2 text-sm text-slate-400">
              An unexpected error occurred. You can try again.
            </p>
            <div className="mt-6 flex gap-3">
              <button type="button" className="ui-button-primary" onClick={() => reset()}>
                Retry
              </button>
              <Link className="ui-button-secondary" href="/">
                Go to login
              </Link>
            </div>
          </div>
        </div>
      </body>
    </html>
  );
}
