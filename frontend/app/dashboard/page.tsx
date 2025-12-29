"use client";

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import PersonaPicker from '@/components/PersonaPicker';

export default function Dashboard() {
    const router = useRouter();
    const [username] = useState(() => {
        if (typeof window === 'undefined') return '';
        return localStorage.getItem('username') || '';
    });
    const [pickerOpen, setPickerOpen] = useState(false);

    const openNewCallPicker = () => setPickerOpen(true);
    const newPersona = () => {
        router.push(`/dashboard/personas`);
    };

    const handleConfirmPersona = async (slug: string) => {
        // Create room on backend with persona attached
        try {
            const getCookie = (name: string) => {
                const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'))
                return match ? decodeURIComponent(match[2]) : null
            }

            const csrf = getCookie('csrftoken')

            const res = await fetch('http://localhost:8000/api/rooms/', {
                method: 'POST',
                credentials: 'include',
                headers: { 'Content-Type': 'application/json', ...(csrf ? { 'X-CSRFToken': csrf } : {}) },
                body: JSON.stringify({ name: `Call with ${slug}`, persona: slug })
            });

            if (!res.ok) {
                const body = await res.text();
                alert('Failed to create room: ' + (body || res.statusText));
                setPickerOpen(false);
                return;
            }

            const data = await res.json();
            const roomId = data.id;
            setPickerOpen(false);
            router.push(`/room/${roomId}?persona=${encodeURIComponent(slug)}`);

        } catch (e) {
            alert('Failed to create room: ' + e);
            setPickerOpen(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-950 text-slate-50">
            <div className="ui-container py-10">
                <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between mb-10">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
                        <p className="mt-1 text-sm text-slate-400">Start a call, or manage personas.</p>
                    </div>
                </header>

                <main>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <button
                            type="button"
                            className="ui-surface text-left p-6 hover:bg-slate-900/60 transition-colors cursor-pointer"
                            onClick={openNewCallPicker}
                        >
                            <div className="h-12 w-12 bg-indigo-500/20 rounded-2xl flex items-center justify-center mb-4 border border-indigo-500/20">
                                <svg className="w-6 h-6 text-indigo-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                            </div>
                            <h2 className="text-xl font-semibold mb-2">New call</h2>
                            <p className="text-slate-400">Start a one-on-one video call with the selected AI persona.</p>
                        </button>

                        <button
                            type="button"
                            className="ui-surface text-left p-6 hover:bg-slate-900/60 transition-colors cursor-pointer"
                            onClick={newPersona}
                        >
                            <div className="h-12 w-12 bg-teal-500/20 rounded-2xl flex items-center justify-center mb-4 border border-teal-500/20">
                                <svg className="w-6 h-6 text-teal-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
                            </div>
                            <h2 className="text-xl font-semibold mb-2">Manage personas</h2>
                            <p className="text-slate-400">Edit system prompts, voice settings, and behavior profiles.</p>
                        </button>
                    </div>
                </main>
            </div>

            <PersonaPicker open={pickerOpen} onClose={() => setPickerOpen(false)} onConfirm={handleConfirmPersona} />
        </div>
    );
}
