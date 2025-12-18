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

    if (loading) return <div className="p-8 text-center text-gray-500">Loading Personas...</div>

    return (
        <div className="p-8 max-w-6xl mx-auto">
            <div className="flex justify-between items-center mb-8">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900 dark:text-white">AI Personas</h1>
                    <p className="text-gray-500 mt-1">Manage and design your AI assistants.</p>
                </div>
                <Link
                    href="/dashboard/personas/new"
                    className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors"
                >
                    Create New
                </Link>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {personas.map(persona => (
                    <div
                        key={persona.uuid}
                        className="group block bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg transition-all"
                    >
                        <div className="flex justify-between items-start mb-4">
                            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-xl">
                                {persona.display_name[0]}
                            </div>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${persona.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>
                                {persona.is_active ? 'Active' : 'Inactive'}
                            </span>
                        </div>

                        <h3 className="text-xl font-semibold mb-2 group-hover:text-blue-600 transition-colors">
                            <Link href={`/dashboard/personas/${persona.slug}`} className="hover:underline">{persona.display_name}</Link>
                        </h3>

                        <p className="text-gray-500 text-sm line-clamp-3 mb-4 h-15">
                            {persona.description_text || "No description provided."}
                        </p>

                        <div className="flex justify-between items-center text-xs text-gray-400 border-t pt-4 dark:border-gray-700">
                            <div>
                                <span className="mr-4">v{persona.version}</span>
                                <span>{persona.slug}</span>
                            </div>

                            <div className="flex items-center gap-3">
                                <Link href={`/dashboard/personas/${persona.slug}`} className="text-blue-600 hover:underline text-sm">View</Link>
                                <button
                                    onClick={() => handleDelete(persona.slug)}
                                    disabled={deleting === persona.slug}
                                    className="text-red-600 hover:text-red-800 text-sm hover:underline cursor-pointer"
                                >
                                    {deleting === persona.slug ? 'Deleting...' : 'Delete'}
                                </button>
                            </div>
                        </div>
                    </div>
                ))}

                {personas.length === 0 && (
                    <div className="col-span-full text-center py-12 bg-gray-50 dark:bg-gray-800/50 rounded-xl border border-dashed border-gray-300">
                        <p className="text-gray-500 mb-4">No personas found.</p>
                        <Link href="/dashboard/personas/new" className="text-blue-600 hover:underline">
                            Create your first persona
                        </Link>
                    </div>
                )}
            </div>
        </div>
    )
}
