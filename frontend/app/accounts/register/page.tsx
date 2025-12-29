'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { register } from '../../lib/auth';

export default function RegisterPage() {
    const [username, setUsername] = useState('');
    const [password1, setPassword1] = useState('');
    const [password2, setPassword2] = useState('');
    const [error, setError] = useState('');
    const [errors, setErrors] = useState<{ username?: string; password?: string; password2?: string }>({});
    const [loading, setLoading] = useState(false);
    const [showPassword, setShowPassword] = useState(false);
    const router = useRouter();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setError('');
        setErrors({});

        if (password1 !== password2) {
            setErrors({ password2: 'Passwords do not match' });
            return;
        }

        setLoading(true);

        try {
            const result = await register(username, password1, password2);

            if (result.success) {
                // Redirect to dashboard (user is auto-logged in)
                router.push(result.redirectUrl || '/dashboard');
            } else {
                // Handle field-specific errors
                if (result.errors) {
                    setErrors(result.errors);

                    // Clear only the incorrect field(s)
                    if (result.errors.username) {
                        setUsername('');
                    }
                    if (result.errors.password) {
                        setPassword1('');
                        setPassword2('');
                    }
                    if (result.errors.password2) {
                        setPassword2('');
                    }
                } else {
                    // Generic error
                    setError(result.error || 'Registration failed');
                }
            }
        } catch (err) {
            setError('An unexpected error occurred');
        } finally {
            setLoading(false);
        }
    };

    const getPasswordStrength = () => {
        if (!password1) return 0;
        let strength = 0;
        if (password1.length > 5) strength += 1;
        if (password1.length > 8) strength += 1;
        if (/[A-Z]/.test(password1)) strength += 1;
        if (/[0-9]/.test(password1)) strength += 1;
        if (/[^A-Za-z0-9]/.test(password1)) strength += 1;
        return strength;
    };

    const strength = getPasswordStrength();
    const strengthColor = strength < 2 ? '#ef4444' : strength < 4 ? '#eab308' : '#10b981';
    const strengthWidth = password1 ? `${(strength / 5) * 100}%` : '0%';

    return (
        <div className="relative flex min-h-screen items-center justify-center p-4 sm:p-8 bg-slate-950 text-slate-50">
            {/* Background Elements */}
            <div className="fixed inset-0 overflow-hidden -z-10 pointer-events-none">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-500/20 rounded-full blur-[100px] animate-pulse"></div>
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-teal-500/10 rounded-full blur-[100px]"></div>
            </div>

            {/* Main Card */}
            <main className="ui-surface-strong w-full max-w-4xl grid md:grid-cols-2 overflow-hidden min-h-[600px] shadow-2xl shadow-black/40">

                {/* Left Panel: Brand / Visual */}
                <div className="hidden md:flex flex-col justify-between p-10 relative bg-gradient-to-br from-indigo-500/15 via-slate-900/10 to-slate-950/10">
                    <div className="z-10">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="w-10 h-10 rounded-xl flex items-center justify-center shadow-lg bg-gradient-to-br from-indigo-500 to-teal-500">
                                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
                            </div>
                            <span className="text-2xl font-bold tracking-tight text-white">VideoConf AI</span>
                        </div>
                        <h1 className="text-4xl font-bold text-white mb-4 leading-tight">Join the Future.</h1>
                        <p className="text-slate-300 text-lg leading-relaxed">Create your account and start building meaningful AI personas for your conversations.</p>
                    </div>

                    <div className="z-10 mt-auto pt-10 text-sm text-slate-400">
                        &copy; {new Date().getFullYear()} VideoConf AI
                    </div>
                </div>

                {/* Right Panel: Form */}
                <div className="p-8 sm:p-12 flex flex-col justify-center relative bg-slate-900/30">
                    <div className="w-full max-w-sm mx-auto">
                        <div className="mb-8 text-center md:text-left">
                            <h2 className="text-3xl font-bold text-white mb-2">Create Account</h2>
                            <p className="text-slate-400">Join to start building your AI personas.</p>
                        </div>

                        {error && (
                            <div className="p-4 mb-6 rounded-xl text-sm font-medium border border-red-500/30 bg-red-500/10 text-red-200">
                                {error}
                            </div>
                        )}

                        <form onSubmit={handleSubmit} className="space-y-5">
                            <div className="relative group">
                                <label htmlFor="username" className="absolute left-4 top-3 text-slate-400 text-sm transition-all pointer-events-none peer-focus:text-xs peer-focus:-translate-y-2 peer-focus:text-indigo-400">
                                    Username
                                </label>
                                <input
                                    type="text"
                                    id="username"
                                    value={username}
                                    onChange={(e) => setUsername(e.target.value)}
                                    required
                                    className={`peer w-full pt-6 pb-2 px-4 rounded-xl border transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:border-indigo-400 bg-slate-900/60 text-slate-50 ${errors.username ? 'border-red-500/60' : 'border-white/10'}`}
                                />
                                {errors.username && (
                                    <p className="text-red-300 text-sm mt-1">{errors.username}</p>
                                )}
                            </div>

                            <div className="relative group">
                                <label htmlFor="password1" className="absolute left-4 top-3 text-slate-400 text-sm transition-all pointer-events-none peer-focus:text-xs peer-focus:-translate-y-2 peer-focus:text-indigo-400">
                                    Password
                                </label>
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    id="password1"
                                    value={password1}
                                    onChange={(e) => setPassword1(e.target.value)}
                                    required
                                    className={`peer w-full pt-6 pb-2 px-4 pr-10 rounded-xl border transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:border-indigo-400 bg-slate-900/60 text-slate-50 ${errors.password ? 'border-red-500/60' : 'border-white/10'}`}
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-3 text-slate-400 hover:text-slate-100 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 rounded-md"
                                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                    </svg>
                                </button>
                                {errors.password && (
                                    <p className="text-red-300 text-sm mt-1">{errors.password}</p>
                                )}
                                {/* Password Strength Meter */}
                                {password1 && (
                                    <div className="mt-2 h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                                        <div
                                            className="h-full transition-all duration-300"
                                            style={{
                                                width: strengthWidth,
                                                backgroundColor: strengthColor,
                                            }}
                                        />
                                    </div>
                                )}
                            </div>

                            <div className="relative group">
                                <label htmlFor="password2" className="absolute left-4 top-3 text-slate-400 text-sm transition-all pointer-events-none peer-focus:text-xs peer-focus:-translate-y-2 peer-focus:text-indigo-400">
                                    Confirm Password
                                </label>
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    id="password2"
                                    value={password2}
                                    onChange={(e) => setPassword2(e.target.value)}
                                    required
                                    className={`peer w-full pt-6 pb-2 px-4 rounded-xl border transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500/50 focus-visible:border-indigo-400 bg-slate-900/60 text-slate-50 ${errors.password2 ? 'border-red-500/60' : 'border-white/10'}`}
                                />
                                {errors.password2 && (
                                    <p className="text-red-300 text-sm mt-1">{errors.password2}</p>
                                )}
                            </div>

                            <div className="text-xs text-slate-400 mt-2">
                                By creating an account, you agree to our <a href="#" className="text-indigo-300 hover:underline">Terms of Service</a> and <a href="#" className="text-indigo-300 hover:underline">Privacy Policy</a>.
                            </div>

                            <button
                                type="submit"
                                disabled={loading}
                                className="ui-button-primary w-full py-3"
                            >
                                {loading ? 'Creating account...' : 'Create account'}
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                </svg>
                            </button>
                        </form>

                        <div className="mt-8 text-center text-sm text-slate-400">
                            Already have an account?{' '}
                            <Link href="/" className="font-medium text-indigo-300 hover:text-indigo-200 transition-colors">
                                Sign in
                            </Link>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}
