"use client"

import React, { useEffect, useState } from 'react'
import Link from 'next/link'

export interface PersonaBrief {
    uuid: string
    slug: string
    display_name: string
    description_text: string
    is_active: boolean
    version: number
}

export default function PersonaPicker({ open, onClose, onConfirm }: { open: boolean, onClose: () => void, onConfirm: (slug: string) => void }) {
    const [personas, setPersonas] = useState<PersonaBrief[]>([])
    const [loading, setLoading] = useState(false)
    const [selected, setSelected] = useState<string | null>(null)

    useEffect(() => {
        if (!open) return
        setLoading(true)
        fetch('http://localhost:8000/api/personas/', { credentials: 'include' })
            .then(r => r.json())
            .then(data => setPersonas(data))
            .catch(() => {})
            .finally(() => setLoading(false))
    }, [open])

    if (!open) return null

    return (
        <div className="fixed inset-0 z-50 flex items-end md:items-center justify-center">
            <div className="absolute inset-0 bg-black/60" onClick={onClose} />

            <div className="relative bg-gray-900 text-white rounded-t-lg md:rounded-lg w-full md:w-3/4 lg:w-2/3 max-h-[80vh] overflow-auto p-6 z-10">
                <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold">Choose an AI Persona</h2>
                    <button onClick={onClose} className="text-gray-400 hover:text-white">Close</button>
                </div>

                {loading && <div className="p-6 text-center text-gray-400">Loading personas...</div>}

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {personas.map(p => (
                        <button
                            key={p.uuid}
                            onClick={() => setSelected(p.slug)}
                            className={`group p-4 rounded-lg text-left border ${selected === p.slug ? 'border-blue-500 ring-2 ring-blue-500' : 'border-gray-700'} bg-gray-800 hover:shadow-lg transition`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-lg">{p.display_name[0]}</div>
                                <span className={`px-2 py-1 rounded text-xs font-medium ${p.is_active ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-600'}`}>{p.is_active ? 'Active' : 'Inactive'}</span>
                            </div>
                            <h3 className="font-semibold mb-1">{p.display_name}</h3>
                            <p className="text-sm text-gray-300 mb-3 line-clamp-3">{p.description_text || 'No description provided.'}</p>
                            <div className="flex justify-between items-center text-xs text-gray-400">
                                <span>v{p.version}</span>
                                <span className="text-xs text-gray-400">{p.slug}</span>
                            </div>
                        </button>
                    ))}
                </div>

                <div className="mt-6 flex justify-end gap-3">
                    <button onClick={onClose} className="px-4 py-2 rounded bg-gray-700 hover:bg-gray-600">Cancel</button>
                    <button
                        onClick={() => selected && onConfirm(selected)}
                        disabled={!selected}
                        className="px-4 py-2 rounded bg-blue-600 hover:bg-blue-700 disabled:opacity-50"
                    >
                        {selected ? `Start Call with ${personas.find(p => p.slug === selected)?.display_name || 'Persona'}` : 'Select a persona'}
                    </button>
                </div>
            </div>
        </div>
    )
}
