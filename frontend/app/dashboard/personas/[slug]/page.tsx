"use client"

import { useEffect, useState, useRef } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { merge } from 'lodash'
import { getCSRFToken } from '@/app/lib/auth'

// --- Constants & Types ---

const PRESETS = [
    {
        id: "helpful-assistant",
        label: "Helpful Assistant",
        desc: "A general-purpose assistant that provides clear, accurate, and concise help across a wide range of topics. It focuses on understanding the user's intent quickly and delivering practical, easy-to-follow responses without unnecessary verbosity.",
        system_prompt: "You are a helpful general-purpose assistant. You listen carefully to the user's request, identify the core intent, and respond with clear, accurate, and concise information. You avoid unnecessary details, do not ramble, and prioritize practical usefulness. When a request is ambiguous, you ask a brief clarifying question. Your tone is polite, neutral, and professional, and your responses are well-structured and easy to understand when spoken aloud."
    },
    {
        id: "technical-expert",
        label: "Technical Expert",
        desc: "A senior software engineer who provides in-depth technical explanations, evaluates trade-offs, and helps users solve complex engineering problems by asking thoughtful clarifying questions and reasoning step by step.",
        system_prompt: "You are a senior software engineer with deep technical expertise. You approach problems methodically by first understanding the user's context and constraints. When necessary, you ask precise clarifying questions before proposing solutions. You explain concepts step by step, use correct technical terminology, and discuss trade-offs and best practices. You avoid speculation, do not oversimplify complex topics, and tailor the depth of your explanations to the user's apparent level. Your tone is calm, confident, and instructive, and your responses are structured and clear when spoken aloud."
    },
    {
        id: "empathetic-coach",
        label: "Empathetic Coach",
        desc: "A compassionate life coach who listens attentively, acknowledges emotions, and offers thoughtful, non-judgmental guidance to help users reflect and move forward at their own pace.",
        system_prompt: "You are an empathetic life coach. You listen carefully and acknowledge the user's feelings before offering guidance. You validate emotions without judgment, reflect what you hear to show understanding, and ask gentle, open-ended questions that encourage self-reflection. You avoid giving prescriptive advice too quickly and never dismiss or minimize the user's experience. Your tone is warm, calm, and supportive, and your responses are measured, reassuring, and well-suited for voice-based conversation."
    },
    {
        id: "technical-interviewer",
        label: "Technical Interviewer",
        desc: "A professional interviewer who conducts structured technical interviews, evaluates answers fairly, and guides the candidate through progressively challenging questions.",
        system_prompt: "You are a senior technical interviewer. You guide the conversation, ask structured questions, evaluate answers objectively, request clarification or retries when needed, and conclude with a concise assessment of the candidate’s performance. You lead the interview and do not wait for the candidate to drive the conversation."
    }
]

// Sarvam AI Voices (fixed list)
const VOICE_PRESETS = [
    { id: "anushka", label: "Anushka (Female)", gender: "Female", description: "Clear Indian English female voice" },
    { id: "manisha", label: "Manisha (Female)", gender: "Female", description: "Warm Indian English female voice" },
    { id: "vidya", label: "Vidya (Female)", gender: "Female", description: "Balanced Indian English female voice" },
    { id: "arya", label: "Arya (Female)", gender: "Female", description: "Neutral Indian English female voice" },
    { id: "abhilash", label: "Abhilash (Male)", gender: "Male", description: "Clear Indian English male voice" },
    { id: "karun", label: "Karun (Male)", gender: "Male", description: "Warm Indian English male voice" },
    { id: "hitesh", label: "Hitesh (Male)", gender: "Male", description: "Neutral Indian English male voice" },
]

type VoiceConfig = {
    provider: string
    preset_id: string | null
    voice_id?: string | null
    language_code?: string | null
    speed: number
    pitch: number
    override?: boolean
    style?: string | null
}

type BehaviorConfig = {
    max_speech_time_s: number
    verbosity: "low" | "default" | "high"
    follow_up_questions: boolean
}

const isAbortError = (err: unknown): boolean => {
    return (
        typeof err === 'object' &&
        err !== null &&
        'name' in err &&
        (err as { name?: unknown }).name === 'AbortError'
    )
}

type PersonaConfig = {
    display_name: string
    slug: string
    greeting: string
    system_prompt: string
    examples: { role: string, text: string }[]
    behavior: BehaviorConfig
    voice: VoiceConfig
    moderation: { enabled: boolean, level: string }
    should_tts: boolean
    metadata: { source_template_id?: string | null }
}

type PersonaFormState = {
    slug: string
    is_active: boolean
    description_text: string
    template_id?: string | null
    config: PersonaConfig
}

const DEFAULT_CONFIG: PersonaConfig = {
    display_name: "",
    slug: "",
    greeting: "Hello!",
    system_prompt: "",
    examples: [],
    behavior: { max_speech_time_s: 45, verbosity: "default", follow_up_questions: false },
    voice: { provider: "sarvam", preset_id: "anushka", voice_id: "anushka", language_code: "en-IN", speed: 1.0, pitch: 0.0, style: "conversational", override: false },
    moderation: { enabled: true, level: "moderate" },
    should_tts: true,
    metadata: { source_template_id: null }
}

export default function PersonaEditor() {
    const router = useRouter()
    const params = useParams()
    const slugParam = Array.isArray(params.slug) ? params.slug[0] : params.slug;
    const isNew = slugParam === 'new'

    // --- State ---

    const [formState, setFormState] = useState<PersonaFormState>({
        slug: '',
        is_active: true,
        description_text: '',
        template_id: null,
        config: DEFAULT_CONFIG
    })

    const [loading, setLoading] = useState(!isNew)
    const [generating, setGenerating] = useState(false)
    const [saving, setSaving] = useState(false)
    const [error, setError] = useState('')

    // Voice Preview
    const [previewingVoice, setPreviewingVoice] = useState(false)
    const previewAudioRef = useRef<HTMLAudioElement>(null)

    // --- Load Data ---

    useEffect(() => {
        if (!isNew && slugParam) {
            fetch(`http://localhost:8000/api/personas/${slugParam}/`)
                .then(res => {
                    if (!res.ok) throw new Error("Failed to load")
                    return res.json()
                })
                .then(data => {
                    // Normalize loaded config with defaults
                    const loadedConfig = merge({}, DEFAULT_CONFIG, data.config)
                    setFormState({
                        slug: data.slug,
                        is_active: data.is_active,
                        description_text: data.description_text || "",
                        // If metadata has source_template_id use it, otherwise try to match description?
                        template_id: loadedConfig.metadata?.source_template_id || null,
                        config: loadedConfig
                    })
                    setLoading(false)
                })
                .catch(err => {
                    setError(err.message)
                    setLoading(false)
                })
        }
    }, [isNew, slugParam])

    // --- Actions ---

    const updateConfig = (updates: Partial<PersonaConfig>) => {
        setFormState(prev => ({
            ...prev,
            config: { ...prev.config, ...updates }
        }))
    }

    const updateVoice = (updates: Partial<VoiceConfig>) => {
        setFormState(prev => ({
            ...prev,
            config: {
                ...prev.config,
                voice: { ...prev.config.voice, ...updates }
            }
        }))
    }

    const handleTemplateSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
        const val = e.target.value
        if (!val) {
            setFormState(s => ({ ...s, template_id: null }))
            return
        }

        const preset = PRESETS.find(p => p.id === val)
        if (preset) {
            setFormState(s => ({
                ...s,
                template_id: preset.id,
                description_text: preset.desc,
                config: {
                    ...s.config,
                    system_prompt: preset.system_prompt,  // Immediately apply the preset's system prompt
                    display_name: preset.label,            // Also update the display name
                }
            }))
        } else {
            // Custom handling
            setFormState(s => ({ ...s, template_id: val }))
        }
    }

    // --- Feature: Smart Merge Generation ---
    const handleGenerateConfig = async () => {
        if (!formState.description_text) return
        setGenerating(true)
        setError('')

        // Identify fields we want to preserve if they are already set by user
        // For MVP, we pass "voice" and "display_name" if slug is not new?
        // Actually, we want to preserve voice if user modified it.
        // Let's passed keys of things we want to check for preservation.

        try {
            const payload = {
                description_text: formState.description_text,
                template_id: formState.template_id,
                current_voice: formState.config.voice,
                preserve_fields: ["voice", "display_name"] // We explicitly ask to respect these
            }

            const res = await fetch('http://localhost:8000/api/personas/convert/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            })

            if (!res.ok) {
                const err = await res.json()
                throw new Error(err.details || err.error || "Generation failed")
            }

            const data = await res.json()
            const suggestedConfig = data.persona_config as PersonaConfig

            // Client-side Smart Merge
            // 1. Merge suggested into current, but respect strict rules
            // 2. Voice: Only overwrite if suggestedConfig.voice.override is TRUE

            setFormState(prev => {
                const current = prev.config

                // Deep merge everything first
                const nextConfig = merge({}, current, suggestedConfig)

                // Restore voice if NO override requested
                if (!suggestedConfig.voice?.override) {
                    // Keep user's voice settings (preset, speed, pitch)
                    // But maybe we want to accept 'style' suggestions?
                    // For safety, let's keep the whole voice object if not overridden
                    nextConfig.voice = {
                        ...current.voice,
                        // we might admit new keys if they were missing, but keep values
                    }
                }

                // Keep display_name if set and not empty, unless override?
                // The requirement says "do not overwrite... unless LLM requests".
                // Since our schema doesn't have display_name override flag, we use heuristic:
                // If user typed a custom name, keep it? 
                // For now let's accept LLM name if it's "New Persona" currently.
                if (current.display_name && current.display_name !== "New Assistant" && current.display_name !== "") {
                    nextConfig.display_name = current.display_name
                }

                return {
                    ...prev,
                    config: nextConfig
                }
            })

        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e))
        } finally {
            setGenerating(false)
        }
    }

    const handleSave = async () => {
        setSaving(true)
        try {
            // Generate slug from display_name if not set
            let slug = isNew ? '' : (formState.slug || formState.config.slug || slugParam)
            if (!slug && formState.config.display_name) {
                const baseSlug = formState.config.display_name
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, '-')
                    .replace(/^-+|-+$/g, '')

                // For new personas, add timestamp to ensure uniqueness
                if (isNew) {
                    const timestamp = Date.now().toString(36) // Convert to base36 for shorter string
                    slug = `${baseSlug}-${timestamp}`
                } else {
                    slug = baseSlug
                }
            }

            if (!slug) {
                setError("Please provide a display name")
                setSaving(false)
                return
            }

            // Get CSRF token
            const csrfToken = await getCSRFToken()

            const payload = {
                slug: slug,
                display_name: formState.config.display_name,
                description_text: formState.description_text,
                is_active: formState.is_active,
                config: formState.config,
            }

            const method = isNew ? 'POST' : 'PUT'
            const url = isNew
                ? 'http://localhost:8000/api/personas/'
                : `http://localhost:8000/api/personas/${slugParam}/`

            const res = await fetch(url, {
                method,
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                },
                credentials: 'include',
                body: JSON.stringify(payload)
            })

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}))
                const errorMsg = errorData.slug
                    ? `Slug error: ${errorData.slug.join(', ')}`
                    : errorData.detail || errorData.error || JSON.stringify(errorData) || "Save failed"
                throw new Error(errorMsg)
            }

            if (isNew) {
                const created = await res.json()
                router.push(`/dashboard/personas/${created.slug}`)
            } else {
                alert("Saved!")
            }

        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e))
        } finally {
            setSaving(false)
        }
    }


    const handleVoicePreview = async () => {
        if (isNew) return alert("Save first")
        setPreviewingVoice(true)

        try {
            // Add timeout controller (3 minutes max for XTTS v2 first synthesis)
            const controller = new AbortController()
            const timeoutId = setTimeout(() => controller.abort(), 180000) // 3 min timeout

            const res = await fetch(`http://localhost:8000/api/personas/${slugParam}/preview/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: "Hello there. This is how I sound in a real conversation. Let me know if this feels natural to you.",
                    voice_config: formState.config.voice
                }),
                signal: controller.signal
            })

            clearTimeout(timeoutId)

            if (res.status === 429) {
                const data = await res.json()
                throw new Error(data.error || "Synthesis in progress. Please wait and try again.")
            }

            if (!res.ok) {
                const errorData = await res.json().catch(() => ({}))
                throw new Error(errorData.error || "Preview failed. Check backend logs for details.")
            }

            const blob = await res.blob()
            const url = URL.createObjectURL(blob)
            if (previewAudioRef.current) {
                previewAudioRef.current.src = url
                previewAudioRef.current.play()
            }
        } catch (e: unknown) {
            if (isAbortError(e)) {
                alert("Preview timed out after 3 minutes. The system may be overloaded. Try restarting the backend or switching to CPU mode.")
            } else {
                const message = e instanceof Error ? e.message : String(e)
                alert(`Preview failed: ${message}`)
            }
        } finally {
            setPreviewingVoice(false)
        }
    }

    // --- Render ---

    if (loading) return <div className="p-10 text-center">Loading Persona...</div>

    // Determine derived template value for dropdown
    const currentTemplateValue = formState.template_id || (PRESETS.find(p => p.desc === formState.description_text)?.id) || "__custom__"

    return (
        <div className="max-w-5xl mx-auto p-6 space-y-8 pb-32">
            {/* Header */}
            <div className="rounded-lg flex justify-between items-center sticky top-0 bg-white/80 dark:bg-black/80 backdrop-blur z-10 py-4 border-b">
                <div>
                    <h1 className="ml-3 text-2xl font-bold">{formState.config.display_name || "New Persona"}</h1>
                    <p className="ml-3 text-sm text-gray-500">ID: {slugParam}</p>
                </div>
                <div className="flex gap-2">
                    <button className="px-4 py-2 border rounded cursor-pointer hover:bg-gray-800" onClick={() => router.push('/dashboard/personas')}>Go Back</button>
                    <button
                        onClick={handleSave}
                        disabled={saving}
                        className="px-6 py-2 bg-black text-white rounded hover:bg-gray-800 disabled:opacity-50 cursor-pointer mr-3"
                    >
                        {saving ? "Saving..." : "Save Persona"}
                    </button>
                </div>
            </div>

            {error && <div className="bg-red-50 text-red-600 p-4 rounded border-red-200 border">{error}</div>}

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* LEFT COLUMN: Design & Identity */}
                <div className="lg:col-span-2 space-y-6">

                    {/* 1. Identity */}
                    <section className="bg-white dark:bg-gray-800 p-6 rounded-xl border">
                        <h3 className="font-semibold text-lg mb-4">Identity</h3>
                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium mb-1">Display Name</label>
                                <input
                                    className="w-full p-2 border rounded dark:bg-gray-900 dark:border-gray-700 dark:text-white"
                                    value={formState.config.display_name}
                                    onChange={e => updateConfig({ display_name: e.target.value })}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium mb-1">Response Greeting</label>
                                <input
                                    className="w-full p-2 border rounded dark:bg-gray-900 dark:border-gray-700 dark:text-white"
                                    value={formState.config.greeting}
                                    onChange={e => updateConfig({ greeting: e.target.value })}
                                />
                            </div>
                        </div>
                    </section>

                    {/* 2. Generation / Template */}
                    <section className="bg-white dark:bg-gray-800 p-6 rounded-xl border relative overflow-hidden">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-semibold text-lg">Persona Design</h3>
                            <select
                                value={currentTemplateValue}
                                onChange={handleTemplateSelect}
                                className="p-2 border rounded text-sm bg-gray-50 dark:bg-gray-700 dark:border-gray-600 dark:text-white max-w-[200px]"
                            >
                                <option value="" disabled>Select a template...</option>
                                <option value="__custom__">Custom</option>
                                {PRESETS.map(p => <option key={p.id} value={p.id}>{p.label}</option>)}
                            </select>
                        </div>

                        <textarea
                            className="w-full h-32 p-3 border rounded text-sm mb-4 font-mono bg-gray-50 dark:bg-gray-900 dark:border-gray-700 dark:text-gray-200 dark:placeholder-gray-500"
                            placeholder="Describe the persona's personality, role, and tone..."
                            value={formState.description_text}
                            onChange={e => setFormState(s => ({ ...s, description_text: e.target.value, template_id: null }))} // Switching to custom on edit
                        />

                        <button
                            onClick={handleGenerateConfig}
                            disabled={generating || !formState.description_text}
                            className="w-full bg-blue-600 text-white py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
                        >
                            {generating ? "✨ Generating Configuration (Smart Merge)..." : "✨ Generate Configuration"}
                        </button>
                    </section>

                    {/* 3. System Prompt (Read Only / Advanced) */}
                    <section className="bg-gray-50 dark:bg-gray-900 p-6 rounded-xl border">
                        <h3 className="font-semibold text-sm text-gray-500 uppercase tracking-wider mb-2">Generated System Prompt</h3>
                        <textarea
                            readOnly
                            className="w-full h-48 p-3 text-xs font-mono border rounded bg-white text-gray-600 dark:bg-gray-950 dark:border-gray-800 dark:text-gray-400"
                            value={formState.config.system_prompt}
                        />
                        <p className="text-xs text-gray-400 mt-2">
                            This prompt is auto-generated based on the configuration. Edit schema fields to update it.
                        </p>
                    </section>
                </div>

                {/* RIGHT COLUMN: Voice & Testing */}
                <div className="space-y-6">

                    {/* Voice Panel */}
                    <section className="bg-white dark:bg-gray-800 p-6 rounded-xl border">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="font-semibold text-lg">Voice Settings</h3>
                            {formState.config.voice.override &&
                                <span className="text-xs bg-yellow-100 text-yellow-800 px-2 py-1 rounded">Overridden by AI</span>
                            }
                        </div>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-1">Voice Preset</label>
                                <select
                                    className="w-full p-2 border rounded dark:bg-gray-900 dark:border-gray-700 dark:text-white"
                                    value={formState.config.voice.preset_id || 'anushka'}
                                    onChange={e => {
                                        const preset = VOICE_PRESETS.find(v => v.id === e.target.value);
                                        updateVoice({ 
                                            preset_id: preset?.id || null,
                                            voice_id: preset?.id || null
                                        });
                                    }}
                                >
                                    {VOICE_PRESETS.map(v => (
                                        <option key={v.id} value={v.id}>{v.label}</option>
                                    ))}
                                </select>
                                {VOICE_PRESETS.find(v => v.id === (formState.config.voice.preset_id || 'anushka'))?.description && (
                                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 italic">
                                        {VOICE_PRESETS.find(v => v.id === (formState.config.voice.preset_id || 'anushka'))?.description}
                                    </p>
                                )}
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">Speed: {formState.config.voice.speed?.toFixed(1)}x</label>
                                    <input
                                        type="range" min="0.5" max="2.0" step="0.1"
                                        className="w-full accent-blue-600"
                                        value={formState.config.voice.speed}
                                        onChange={e => updateVoice({ speed: parseFloat(e.target.value) })}
                                    />
                                    <p className="text-xs text-gray-400 mt-1">supports 0.5x-2.0x pace</p>
                                </div>
                                <div>
                                    <label className="block text-xs text-gray-500 mb-1">Pitch: {formState.config.voice.pitch?.toFixed(2)} st</label>
                                    <input
                                        type="range" min="-0.75" max="0.75" step="0.05"
                                        className="w-full accent-blue-600"
                                        value={formState.config.voice.pitch}
                                        onChange={e => updateVoice({ pitch: parseFloat(e.target.value) })}
                                    />
                                    <p className="text-xs text-gray-400 mt-1">Range -0.75 to +0.75 semitones</p>
                                </div>
                            </div>

                            <button
                                onClick={handleVoicePreview}
                                disabled={previewingVoice || isNew}
                                className="w-full py-2 border rounded flex justify-center items-center gap-2 hover:bg-gray-50 dark:border-gray-700 dark:hover:bg-gray-800 dark:text-gray-200 disabled:opacity-50 disabled:bg-gray-100 dark:disabled:bg-gray-900 dark:disabled:text-gray-500"
                            >
                                {previewingVoice ? "Loading Audio..." : "▶ Preview Voice"}
                            </button>
                            <audio ref={previewAudioRef} hidden />
                        </div>
                    </section>

                    {/* Test Panel (Legacy / Live) */}
                    <section className="bg-green-50 dark:bg-green-900/10 p-6 rounded-xl border border-green-200">
                        <h3 className="font-semibold text-lg mb-2 text-green-900 dark:text-green-100">Test Interaction</h3>
                        <p className="text-sm text-gray-600 mb-4">Type a message to test response generation + TTS.</p>

                        <TestPanel slug={slugParam} isNew={isNew} voiceConfig={formState.config.voice} />
                    </section>
                </div>
            </div>

        </div>
    )
}

// Sub-component for Testing
function TestPanel({ slug, isNew, voiceConfig }: { slug: string | string[] | undefined, isNew: boolean, voiceConfig: VoiceConfig }) {
    const [text, setText] = useState("Hello!")
    const [testing, setTesting] = useState(false)
    const audioRef = useRef<HTMLAudioElement>(null)

    const runTest = async () => {
        if (isNew) return alert("Save first")
        setTesting(true)
        try {
            const res = await fetch(`http://localhost:8000/api/personas/${slug}/test_tts/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    voice_config: voiceConfig // Use live config from props
                })
            })
            if (res.ok) {
                const blob = await res.blob()
                const url = URL.createObjectURL(blob)
                if (audioRef.current) {
                    audioRef.current.src = url
                    audioRef.current.play()
                }
            }
        } catch (e) { console.error(e) }
        finally { setTesting(false) }
    }

    return (
        <div className="">
            <input
                className="flex-1 p-2 border rounded dark:bg-gray-800 dark:border-gray-700 dark:text-white"
                value={text} onChange={e => setText(e.target.value)}
            />
            <div className="mt-2 ">
                <button
                    onClick={runTest} disabled={testing}
                    className="cursor-pointer bg-green-600 text-white px-4 rounded py-1 hover:bg-green-700 disabled:opacity-50"
                >
                    {testing ? "..." : "Speak"}
                </button>
                <audio ref={audioRef} hidden />
            </div>
        </div>
    )
}
