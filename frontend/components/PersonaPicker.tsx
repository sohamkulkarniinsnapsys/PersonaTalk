"use client"

import React, { useEffect, useRef, useState } from 'react'
import Link from 'next/link'
import { apiFetch } from '@/app/lib/api'
import { Modal } from '@/components/ui/Modal'
import { Button } from '@/components/ui/Button'

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
    const triggerRef = useRef<HTMLElement | null>(null)

    useEffect(() => {
        if (!open) return

        // Capture the element that opened the modal so focus can be restored.
        triggerRef.current = document.activeElement as HTMLElement | null

        setLoading(true)
        apiFetch('/api/personas/')
            .then(r => r.json())
            .then(data => setPersonas(data))
            .catch(() => {})
            .finally(() => setLoading(false))
    }, [open])

    useEffect(() => {
        if (open) return
        // Reset selection after close for a predictable next-open state.
        setSelected(null)
        // Restore focus to the opener, if it still exists.
        triggerRef.current?.focus?.()
        triggerRef.current = null
    }, [open])

    if (!open) return null

    return (
        <Modal open={open} onClose={onClose} title="Choose an AI persona">
            {loading && <div className="p-6 text-center text-slate-300">Loading personas…</div>}

            {!loading && personas.length === 0 && (
                <div className="ui-surface p-6 text-center">
                    <p className="text-sm text-slate-300">No personas found.</p>
                    <p className="mt-1 text-xs text-slate-400">Create one to start a call.</p>
                    <div className="mt-4 flex items-center justify-center gap-3">
                        <Link className="ui-button-primary" href="/dashboard/personas/new">
                            Create persona
                        </Link>
                        <Button type="button" variant="secondary" onClick={onClose}>
                            Close
                        </Button>
                    </div>
                </div>
            )}

            {personas.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {personas.map(p => (
                        <button
                            key={p.uuid}
                            type="button"
                            onClick={() => setSelected(p.slug)}
                            className={`cursor-pointer group ui-surface p-4 text-left transition-colors ${selected === p.slug ? 'ring-2 ring-indigo-500/60' : ''} hover:bg-slate-800`}
                        >
                            <div className="flex justify-between items-start mb-2">
                                <div className="w-10 h-10 rounded-2xl bg-slate-900/70 border border-slate-700 flex items-center justify-center text-slate-100 font-semibold text-lg">
                                    {p.display_name[0]}
                                </div>
                                <span className={`ui-badge ${p.is_active ? 'bg-slate-800/70 text-slate-200 border border-slate-700' : 'bg-slate-900/40 text-slate-400 border border-slate-800'}`}>
                                    {p.is_active ? 'Active' : 'Inactive'}
                                </span>
                            </div>
                            <h3 className="font-semibold mb-1 text-slate-100">{p.display_name}</h3>
                            <p className="text-sm text-slate-400 mb-3 line-clamp-3">{p.description_text || 'No description provided.'}</p>
                            <div className="flex justify-between items-center text-xs text-slate-400">
                                <span>v{p.version}</span>
                                <span className="text-xs text-slate-400">{p.slug}</span>
                            </div>
                        </button>
                    ))}
                </div>
            )}

            {personas.length > 0 && (
                <div className="mt-6 flex justify-end gap-3">
                    <Button type="button" variant="secondary" onClick={onClose} className='cursor-pointer'>
                        Cancel
                    </Button>
                    <Button type="button" variant="primary" onClick={() => selected && onConfirm(selected)} disabled={!selected} className={selected ? 'cursor-pointer' : 'cursor-not-allowed'}>
                        {selected
                            ? `Start call with ${personas.find(p => p.slug === selected)?.display_name || 'persona'}`
                            : 'Select a persona'}
                    </Button>
                </div>
            )}
        </Modal>
    )
}
