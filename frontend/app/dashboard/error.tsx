'use client';

import { useEffect } from 'react';

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('Dashboard route error:', error);
  }, [error]);

  return (
    <div className="ui-container py-10">
      <div className="ui-surface p-8">
        <h1 className="text-xl font-semibold">Dashboard error</h1>
        <p className="mt-2 text-sm text-slate-400">
          Something went wrong while loading this section.
        </p>
        <div className="mt-6">
          <button type="button" className="ui-button-primary" onClick={() => reset()}>
            Retry
          </button>
        </div>
      </div>
    </div>
  );
}
