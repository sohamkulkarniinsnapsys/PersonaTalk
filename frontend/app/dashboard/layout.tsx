'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { checkAuth, logout } from '@/app/lib/auth';

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();

  const [authStatus, setAuthStatus] = useState<'checking' | 'authed' | 'unauthed'>('checking');

  const username = useMemo(() => {
    if (typeof window === 'undefined') return '';
    return localStorage.getItem('username') || '';
  }, []);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      try {
        const ok = await checkAuth();
        if (cancelled) return;
        setAuthStatus(ok ? 'authed' : 'unauthed');
        if (!ok) router.replace('/');
      } catch {
        if (cancelled) return;
        setAuthStatus('unauthed');
        router.replace('/');
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const handleLogout = async () => {
    try {
      await logout();
    } finally {
      try {
        localStorage.removeItem('username');
      } catch {
        // ignore
      }
      router.replace('/');
    }
  };

  if (authStatus === 'checking') {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-50">
        <div className="ui-container py-10">
          <div className="ui-surface p-6 text-sm text-slate-300">Checking session…</div>
        </div>
      </div>
    );
  }

  // If unauthed, we already navigated away.
  if (authStatus === 'unauthed') return null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <header className="border-b border-slate-800 bg-slate-950/40 backdrop-blur">
        <div className="ui-container py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <Link href="/dashboard" className="text-sm font-semibold">
              VideoConf AI
            </Link>
            <div className="text-xs text-slate-400">
              {pathname?.startsWith('/dashboard/personas') ? 'Personas' : 'Dashboard'}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="text-sm text-slate-300">{username ? `Welcome, ${username}` : 'Welcome'}</div>
            <button type="button" onClick={handleLogout} className="ui-button-danger cursor-pointer">
              Logout
            </button>
          </div>
        </div>
      </header>

      <main>{children}</main>
    </div>
  );
}
