'use client';

import React, { useEffect, useRef, use, useState } from 'react';
import { useWebRTC } from '@/app/hooks/useWebRTC';
import { MediaControls } from '@/components/MediaControls';
import { AIAnimation } from '@/components/AIAnimation';
import { UserListening } from '@/components/UserListening';
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
        setAiSpeaking,
        transcript,
        aiPlaybackStartedTurns
    } = useWebRTC({ roomId });

    const localVideoRef = useRef<HTMLVideoElement>(null);
    const remoteAudioRef = useRef<HTMLAudioElement>(null);
    const [isUserSpeaking, setIsUserSpeaking] = useState(false);

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

    // Detect user speaking from local audio
    useEffect(() => {
        if (!localStream || isMuted) {
            setIsUserSpeaking(false);
            return;
        }

        const audioContext = new AudioContext();
        const analyser = audioContext.createAnalyser();
        const microphone = audioContext.createMediaStreamSource(localStream);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        microphone.connect(analyser);
        analyser.fftSize = 256;

        const checkAudioLevel = () => {
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            setIsUserSpeaking(average > 30);
            requestAnimationFrame(checkAudioLevel);
        };

        checkAudioLevel();

        return () => {
            microphone.disconnect();
            analyser.disconnect();
            audioContext.close();
        };
    }, [localStream, isMuted]);

    // Detect AI speaking from remote audio
    useEffect(() => {
        if (!remoteStream) {
            setAiSpeaking(false);
            return;
        }

        const audioContext = new AudioContext();
        const analyser = audioContext.createAnalyser();
        const mediaSource = audioContext.createMediaStreamSource(remoteStream);
        const dataArray = new Uint8Array(analyser.frequencyBinCount);
        
        mediaSource.connect(analyser);
        analyser.fftSize = 256;

        const checkAudioLevel = () => {
            analyser.getByteFrequencyData(dataArray);
            const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
            setAiSpeaking(average > 30);
            requestAnimationFrame(checkAudioLevel);
        };

        checkAudioLevel();

        return () => {
            mediaSource.disconnect();
            analyser.disconnect();
            audioContext.close();
        };
    }, [remoteStream, setAiSpeaking]);

    const showReconnect = ['failed', 'disconnected', 'closed'].includes(connectionStatus);
    const personaName = persona?.display_name || 'AI Persona';

    return (
        <div className="relative h-screen w-full bg-black overflow-hidden flex flex-col items-center justify-center text-white">

            {/* Background / AI Animation (Center) */}
            <div className="absolute inset-0 flex items-center justify-center z-0">
                <div className="text-center">
                    <AIAnimation isSpeaking={aiSpeaking} personaName={personaName} />

                    <p className="text-gray-400 mt-8 capitalize">Status: {connectionStatus}</p>

                    {showReconnect && (
                        <button
                            onClick={reconnect}
                            className="mt-6 px-6 py-2 bg-white text-black font-semibold rounded-full hover:bg-gray-200 transition-colors z-50 pointer-events-auto"
                        >
                            Reconnect
                        </button>
                    )}
                </div>
            </div>

            {/* User Listening Animation */}
            <UserListening isListening={isUserSpeaking} />

            {/* Hidden Remote Audio */}
            <audio ref={remoteAudioRef} autoPlay />

            {/* Local Video (PiP) - Top Left */}
            <div className="absolute top-4 left-4 w-48 h-36 bg-gray-900 rounded-xl overflow-hidden shadow-2xl border border-gray-800 z-10">
                <video
                    ref={localVideoRef}
                    autoPlay
                    playsInline
                    muted
                    className={`w-full h-full object-cover transform scale-x-[-1] ${isVideoOff ? 'hidden' : 'block'}`}
                />
                {isVideoOff && (
                    <div className="flex items-center justify-center w-full h-full">
                        <span className="text-xs text-gray-500">Camera Off</span>
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
            <div className="absolute top-0 right-0 h-full w-96 bg-black/50 backdrop-blur border-l border-gray-800 z-20">
                <TranscriptPanel
                    items={transcript}
                    personaName={personaName}
                    aiSpeaking={aiSpeaking}
                    playbackStartedTurns={aiPlaybackStartedTurns}
                />
            </div>

        </div>
    );
}
