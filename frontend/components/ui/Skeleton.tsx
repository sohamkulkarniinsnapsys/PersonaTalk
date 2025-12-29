import React from 'react';

export function Skeleton({ className = '' }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-white/10 ${className}`.trim()} />;
}
