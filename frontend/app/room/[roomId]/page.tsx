'use client';

import React, { useEffect, useRef, use } from 'react';
import { useWebRTC } from '@/app/hooks/useWebRTC';
import { MediaControls } from '@/components/MediaControls';
import { AIAnimation } from '@/components/AIAnimation';
import { TranscriptPanel } from '@/components/TranscriptPanel';

export default function CallRoom({ params }: { params: Promise<{ roomId: string }> }) {
    const { roomId } = use(params);
    const {
        localStream,
        remoteStream,
        connectionStatus,
        isMuted,
        isVideoOff,
        toggleMute,
        toggleVideo,
        reconnect,
        cleanup,
        persona,
        aiSpeaking,
        userSpeaking, // NEW: Get userSpeaking from backend events instead of audio analysis
        transcript,
        aiPlaybackStartedTurns,
        aiPlaybackCanceledTurns
    } = useWebRTC({ roomId });

    const localVideoRef = useRef<HTMLVideoElement>(null);
    const remoteAudioRef = useRef<HTMLAudioElement>(null);
    
    // Use backend-driven speaking state (userSpeaking from hook)
    // No longer need local state or audio analysis

    // Cleanup on unmount or navigation
    useEffect(() => {
        return () => {
            cleanup();
        };
    }, [cleanup]);

    // Attach Streams to Elements
    useEffect(() => {
        if (localVideoRef.current && localStream) {
            localVideoRef.current.srcObject = localStream;
        }
    }, [localStream]);

    useEffect(() => {
        if (remoteAudioRef.current && remoteStream) {
            remoteAudioRef.current.srcObject = remoteStream;
            // Attempt autoplay
            remoteAudioRef.current.play().catch(e => {
                console.warn("Autoplay blocked:", e);
            });
        }
    }, [remoteStream]);

    // REMOVED: Local audio analysis for user speaking detection
    // Now using backend VAD events (userSpeaking from useWebRTC hook)
    
    // REMOVED: Remote audio analysis for AI speaking detection  
    // Now using backend TTS lifecycle events (aiSpeaking from useWebRTC hook)

    const showReconnect = ['failed', 'disconnected', 'closed'].includes(connectionStatus);
    const personaName = persona?.display_name || 'AI Persona';

    return (
        <div className="relative h-screen w-full bg-slate-950 overflow-hidden flex flex-col items-center justify-center text-slate-50">

            {/* Background / AI Animation (Center) */}
            <div className="absolute inset-0 flex items-center justify-center z-0">
                <div className="text-center">
                    <AIAnimation 
                        isSpeaking={aiSpeaking} 
                        userSpeaking={userSpeaking}
                        personaName={personaName} 
                    />

                    <p className="text-slate-400 mt-8 capitalize">Status: {connectionStatus}</p>

                    {showReconnect && (
                        <button
                            onClick={reconnect}
                            className="mt-6 ui-button-primary z-50 pointer-events-auto"
                        >
                            Reconnect
                        </button>
                    )}
                </div>
            </div>

            {/* Hidden Remote Audio */}
            <audio ref={remoteAudioRef} autoPlay />

            {/* Local Video (PiP) - Top Left */}
            <div className="absolute top-4 left-4 w-48 h-36 bg-slate-900/40 rounded-xl overflow-hidden shadow-2xl border border-slate-800 z-10">
                <video
                    ref={localVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover transform scale-x-[-1] ${isVideoOff ? 'hidden' : 'block'}`}
                />
                {isVideoOff && (
                    <div className="flex items-center justify-center w-full h-full">
                        <span className="text-xs text-slate-400">Camera Off</span>
                    </div>
                )}
            </div>

            {/* Controls */}
            <div className="absolute bottom-8 z-50">
                <MediaControls
                    isMuted={isMuted}
                    isVideoOff={isVideoOff}
                    onToggleMute={toggleMute}
                    onToggleVideo={toggleVideo}
                    onEndCall={cleanup}
                />
            </div>

            {/* Transcript Sidebar - Right */}
            <div className="absolute top-0 right-0 h-full w-96 bg-slate-950/40 backdrop-blur border-l border-slate-800 z-20">
                <TranscriptPanel
                    items={transcript}
                    personaName={personaName}
                    aiSpeaking={aiSpeaking}
                    playbackStartedTurns={aiPlaybackStartedTurns}
                    playbackCanceledTurns={aiPlaybackCanceledTurns}
                />
            </div>

        </div>
    );
}
