"use client"

import { useEffect, useState } from 'react'
import Link from 'next/link'

interface Persona {
    uuid: string
    slug: string
    display_name: string
    description_text: string
    is_active: boolean
    version: number
}

export default function PersonaList() {
    const [personas, setPersonas] = useState<Persona[]>([])
    const [loading, setLoading] = useState(true)
    const [deleting, setDeleting] = useState<string | null>(null)

    useEffect(() => {
        fetch('http://localhost:8000/api/personas/')
            .then(res => res.json())
            .then(data => {
                setPersonas(data)
                setLoading(false)
            })
            .catch(err => setLoading(false))
    }, [])

    // Helper to read CSRF token from cookies (Django default name: csrftoken)
    const getCookie = (name: string) => {
        const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'))
        return match ? decodeURIComponent(match[2]) : null
    }

    const handleDelete = async (slug: string) => {
        if (!confirm('Delete this persona? This action cannot be undone.')) return

        setDeleting(slug)
        try {
            const csrf = getCookie('csrftoken')
            const res = await fetch(`http://localhost:8000/api/personas/${slug}/`, {
                method: 'DELETE',
                credentials: 'include',
                headers: csrf ? { 'X-CSRFToken': csrf } : undefined,
            })

            if (res.ok || res.status === 204) {
                setPersonas(prev => prev.filter(p => p.slug !== slug))
            } else {
                const body = await res.text()
                alert('Delete failed: ' + (body || res.statusText))
            }
        } catch (e) {
            alert('Delete failed: ' + e)
        } finally {
            setDeleting(null)
        }
    }

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 text-slate-50">
                <div className="ui-container py-10">
                    <div className="ui-surface p-6 text-center text-slate-300">Loading personas…</div>
                </div>
            </div>
        )
    }

    return (
        <div className="min-h-screen bg-slate-950 text-slate-50">
            <div className="ui-container py-10">
            <div className="flex flex-col gap-4 sm:flex-row sm:justify-between sm:items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">AI personas</h1>
                    <p className="mt-1 text-sm text-slate-400">Manage and design your AI assistants.</p>
                </div>
                <Link
                    href="/dashboard/personas/new"
                    className="ui-button-primary"
                >
                    Create New
                </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {personas.map(persona => (
                    <div
                        key={persona.uuid}
                        className="group ui-surface p-6"
                    >
                        <div className="flex justify-between items-start mb-4">
                            <div className="w-12 h-12 rounded-2xl bg-slate-900/70 border border-slate-700 flex items-center justify-center text-slate-100 font-semibold text-xl">
                                {persona.display_name[0]}
                            </div>
                            <span className={`ui-badge ${persona.is_active ? 'bg-slate-800/70 text-slate-200 border-slate-700' : 'bg-slate-900/40 text-slate-400 border-slate-800'}`}>
                                {persona.is_active ? 'Active' : 'Inactive'}
                            </span>
                        </div>

                        <h3 className="text-xl font-semibold mb-2">
                            <Link href={`/dashboard/personas/${persona.slug}`} className="hover:underline underline-offset-4">
                                {persona.display_name}
                            </Link>
                        </h3>

                        <p className="text-slate-400 text-sm line-clamp-3 mb-4">
                            {persona.description_text || "No description provided."}
                        </p>

                        <div className="flex justify-between items-center text-xs text-slate-400 border-t border-slate-800 pt-4">
                            <div>
                                <span className="mr-4">v{persona.version}</span>
                                <span>{persona.slug}</span>
                            </div>

                            <div className="flex items-center gap-3">
                                <Link href={`/dashboard/personas/${persona.slug}`} className="text-sm text-slate-200 hover:underline underline-offset-4">View</Link>
                                <button
                                    onClick={() => handleDelete(persona.slug)}
                                    disabled={deleting === persona.slug}
                                    className="text-sm text-red-300 hover:underline underline-offset-4 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
                                >
                                    {deleting === persona.slug ? 'Deleting...' : 'Delete'}
                                </button>
                            </div>
                        </div>
                    </div>
                ))}

                {personas.length === 0 && (
                    <div className="col-span-full ui-surface p-10 text-center border-dashed border-slate-800">
                        <p className="text-slate-400 mb-4">No personas found.</p>
                        <Link href="/dashboard/personas/new" className="text-slate-200 hover:underline underline-offset-4">
                            Create your first persona
                        </Link>
                    </div>
                )}
            </div>
            </div>
        </div>
    )
}
