"use client";

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import PersonaPicker from '@/components/PersonaPicker';

export default function Dashboard() {
    const router = useRouter();
    const [username, setUsername] = useState('');
    const [pickerOpen, setPickerOpen] = useState(false);

    useEffect(() => {
        const user = localStorage.getItem('username');
        if (!user) {
            router.push('/');
        } else {
            setUsername(user);
        }
    }, [router]);

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
        <div className="min-h-screen bg-gray-900 text-white p-8">
            <header className="flex justify-between items-center mb-12">
                <h1 className="text-3xl font-bold">Dashboard</h1>
                <div className="flex items-center gap-4">
                    <span>Welcome, {username}</span>
                    <button
                        onClick={() => { localStorage.removeItem('username'); router.push('/'); }}
                        className="cursor-pointer bg-linear-to-r from-red-500 to-red-700 text-white hover:from-red-500 hover:to-red-700 btn-hover-shine focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-opacity-40 rounded-lg px-4 py-2 text-sm shadow-sm hover:scale-103 active:scale-95 transition-all duration-200 ease-in-out"
                    >
                        Logout
                    </button>
                </div>
            </header>

            <main className="max-w-4xl mx-auto">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 hover:border-blue-500 transition cursor-pointer" onClick={openNewCallPicker}>
                        <div className="h-12 w-12 bg-blue-600 rounded-full flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                        </div>
                        <h2 className="text-xl font-bold mb-2">New Call</h2>
                        <p className="text-gray-400">Start a new one-on-one video call with the AI Persona.</p>
                    </div>

                    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700 hover:border-blue-500 transition cursor-pointer" onClick={newPersona}>
                        <div className="h-12 w-12 bg-purple-600 rounded-full flex items-center justify-center mb-4">
                            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
                        </div>
                        <h2 className="text-xl font-bold mb-2">Manage Personas</h2>
                        <p className="text-gray-400">Configure AI personalities and voice settings.</p>
                    </div>
                </div>
            </main>

            <PersonaPicker open={pickerOpen} onClose={() => setPickerOpen(false)} onConfirm={handleConfirmPersona} />
        </div>
    );
}
