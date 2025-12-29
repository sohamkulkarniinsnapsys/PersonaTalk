import Link from 'next/link';

export default function NotFound() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-50">
      <div className="ui-container py-16">
        <div className="ui-surface p-8">
          <h1 className="text-2xl font-semibold">Page not found</h1>
          <p className="mt-2 text-sm text-slate-400">
            The page you’re looking for doesn’t exist.
          </p>
          <div className="mt-6 flex gap-3">
            <Link className="ui-button-primary" href="/dashboard">
              Go to dashboard
            </Link>
            <Link className="ui-button-secondary" href="/">
              Go to login
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
