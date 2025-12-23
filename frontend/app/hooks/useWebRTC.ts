import { useEffect, useRef, useState, useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';

export interface UseWebRTCProps {
    roomId: string;
}

export interface PersonaInfo {
    slug: string;
    display_name: string;
}

export const useWebRTC = ({ roomId }: UseWebRTCProps) => {
    const [localStream, setLocalStream] = useState<MediaStream | null>(null);
    const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
    const [connectionStatus, setConnectionStatus] = useState<
        'new' | 'connecting' | 'connected' | 'disconnected' | 'failed' | 'closed'
    >('new');
    const [isMuted, setIsMuted] = useState(false);
    const [isVideoOff, setIsVideoOff] = useState(false);
    const [persona, setPersona] = useState<PersonaInfo | null>(null);
    const [aiSpeaking, setAiSpeaking] = useState(false);
    const [userSpeaking, setUserSpeaking] = useState(false); // NEW: Track user speaking from backend
    const [agentState, setAgentState] = useState<'INIT' | 'GREETING' | 'LISTENING' | 'USER_SPEAKING' | 'USER_FINISHED' | 'VALIDATING_UTTERANCE' | 'CLARIFICATION_REQUIRED' | 'THINKING' | 'AI_SPEAKING' | 'WAIT_FOR_USER' | 'PROCESSING_USER' | 'QUESTION' | 'EVALUATION' | 'RETRY' | 'EXPLANATION' | 'SUMMARY' | 'END' | undefined>(undefined);
    // Tracks which AI turns have started audio playback (signaled by backend)
    const [aiPlaybackStartedTurns, setAiPlaybackStartedTurns] = useState<Set<string>>(new Set());
    const [aiPlaybackCanceledTurns, setAiPlaybackCanceledTurns] = useState<Set<string>>(new Set());
    // Global dedupe for transcript messages
    const seenMessageKeysRef = useRef<Set<string>>(new Set());
    const [transcript, setTranscript] = useState<Array<{
        role: 'user' | 'assistant';
        text: string;
        timestamp?: number;
        utteranceId?: string;
        turnId?: string;
        isStreaming?: boolean;
    }>>([]);

    const pc = useRef<RTCPeerConnection | null>(null);
    const ws = useRef<WebSocket | null>(null);
    const localStreamRef = useRef<MediaStream | null>(null);

    // Refs to keep track of senders for replaceTrack
    const audioSenderRef = useRef<RTCRtpSender | null>(null);
    const videoSenderRef = useRef<RTCRtpSender | null>(null);

    const isMounted = useRef(true);

    const cleanup = useCallback(() => {
        console.log('🧹 Cleanup called - closing all connections');
        
        // Close WebSocket
        if (ws.current) {
            try {
                ws.current.close();
            } catch (e) {
                console.error('Error closing WebSocket:', e);
            }
            ws.current = null;
        }
        
        // Close peer connection
        if (pc.current) {
            try {
                pc.current.close();
            } catch (e) {
                console.error('Error closing peer connection:', e);
            }
            pc.current = null;
        }
        
        // Stop all local audio/video tracks
        if (localStreamRef.current) {
            try {
                localStreamRef.current.getTracks().forEach(track => {
                    console.log(`Stopping track: ${track.kind}`);
                    track.stop();
                });
            } catch (e) {
                console.error('Error stopping local tracks:', e);
            }
            localStreamRef.current = null;
        }
        
        // Clear state
        setLocalStream(null);
        setRemoteStream(null);
        audioSenderRef.current = null;
        videoSenderRef.current = null;
        
        console.log('✅ Cleanup complete');
    }, []);

    const initialize = useCallback(async () => {
        if (!roomId) return;

        setConnectionStatus('connecting');

        // Fetch room details to get persona info
        try {
            const res = await fetch(`http://localhost:8000/api/rooms/${roomId}/`, {
                credentials: 'include'  // CRITICAL: Send Django session cookie
            });
            if (res.ok) {
                const roomData = await res.json();
                if (roomData.persona) {
                    setPersona(roomData.persona);
                }
            }
        } catch (e) {
            console.warn('Failed to fetch room details:', e);
        }

        // 1. Initialize Peer Connection
        const peerConnection = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
            ],
        });

        pc.current = peerConnection;

        // 2. Setup WebSocket
        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${protocol}://localhost:8000/ws/signaling/${roomId}/`;
        const websocket = new WebSocket(wsUrl);
        ws.current = websocket;

        websocket.onopen = async () => {
            console.log('WebSocket Connected');
            if (!isMounted.current) return;

            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    audio: true,
                    video: true
                });

                console.log('✅ Media stream acquired:', {
                    audioTracks: stream.getAudioTracks().length,
                    videoTracks: stream.getVideoTracks().length,
                    audioEnabled: stream.getAudioTracks()[0]?.enabled,
                    audioLabel: stream.getAudioTracks()[0]?.label,
                    audioSettings: stream.getAudioTracks()[0]?.getSettings()
                });

                if (!isMounted.current) {
                    stream.getTracks().forEach(t => t.stop());
                    return;
                }

                setLocalStream(stream);
                localStreamRef.current = stream;
                setIsMuted(false);
                setIsVideoOff(false);

                // Add Tracks to PC and store senders
                stream.getTracks().forEach(track => {
                    console.log(`➕ Adding ${track.kind} track:`, {
                        enabled: track.enabled,
                        muted: track.muted,
                        readyState: track.readyState,
                        label: track.label
                    });
                    const sender = peerConnection.addTrack(track, stream);
                    if (track.kind === 'audio') {
                        audioSenderRef.current = sender;
                    } else if (track.kind === 'video') {
                        videoSenderRef.current = sender;
                    }
                });

                // Create Offer
                const offer = await peerConnection.createOffer();
                await peerConnection.setLocalDescription(offer);

                if (websocket.readyState === WebSocket.OPEN) {
                    websocket.send(JSON.stringify({
                        type: 'offer',
                        sdp: offer.sdp,
                        sdpType: 'offer'
                    }));
                }

            } catch (err) {
                console.error('Error accessing media or creating offer:', err);
                if (isMounted.current) setConnectionStatus('failed');
            }
        };

        websocket.onmessage = async (event) => {
            if (!isMounted.current) return;
            try {
                const data = JSON.parse(event.data);

                if (data.type === 'answer') {
                    await peerConnection.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: data.sdp
                    }));
                } else if (data.type === 'ice-candidate') {
                    if (data.candidate) {
                        await peerConnection.addIceCandidate(new RTCIceCandidate({
                            candidate: data.candidate,
                            sdpMid: data.sdpMid,
                            sdpMLineIndex: data.sdpMLineIndex
                        }));
                    }
                } else if (data.type === 'ai_playback') {
                    const status = String(data.status || '').toLowerCase();
                    const tId = typeof data.turnId === 'string' ? data.turnId : undefined;
                    if (tId) {
                        setAiPlaybackStartedTurns(prev => {
                            const next = new Set(prev);
                            if (status === 'start') {
                                next.add(tId);
                                // If playback restarts, clear any canceled marker for this turn
                                setAiPlaybackCanceledTurns(prevCanceled => {
                                    const cleared = new Set(prevCanceled);
                                    cleared.delete(tId);
                                    return cleared;
                                });
                            } else if (status === 'end') {
                                next.delete(tId);
                                setAiPlaybackCanceledTurns(prevCanceled => {
                                    const cleared = new Set(prevCanceled);
                                    cleared.delete(tId);
                                    return cleared;
                                });
                            } else if (status === 'canceled') {
                                next.delete(tId);
                                setAiPlaybackCanceledTurns(prevCanceled => {
                                    const updated = new Set(prevCanceled);
                                    updated.add(tId);
                                    return updated;
                                });
                            }
                            return next;
                        });
                    }
                } else if (data.type === 'transcript') {
                    // Parse and validate role from backend
                    let role: 'user' | 'assistant' = 'user'; // default to user
                    const textContent = String(data.text ?? '');
                    
                    if (data.role === 'assistant') {
                        role = 'assistant';
                    } else if (data.role === 'user') {
                        role = 'user';
                        // CRITICAL: "I didn't catch that" is an assistant message, not user
                        // This happens when STT confidence is too low
                        if (textContent.includes("I didn't catch that") || 
                            textContent.includes("could you please repeat") ||
                            textContent.includes("speak up") ||
                            textContent.includes("too quiet")) {
                            role = 'assistant'; // Override: this is AI asking user to repeat
                            console.log(`🔄 Overriding role to "assistant" for STT error message`);
                        }
                    } else {
                        // Log unexpected role values for debugging
                        console.warn(`⚠️ Unexpected transcript role: "${data.role}", defaulting to "user"`);
                    }
                    
                    const item = {
                        role: role,
                        text: textContent,
                        timestamp: typeof data.timestamp === 'number' ? data.timestamp : Date.now() / 1000,
                        utteranceId: data.utteranceId as string | undefined,
                        turnId: data.turnId as string | undefined,
                        isStreaming: Boolean(data.isPartial), // Only mark streaming when partials are sent
                    };
                    
                    console.log(`📝 Transcript received: role="${role}", turnId="${item.turnId}", text="${item.text.substring(0, 50)}..."`);
                    
                    // Global, strong dedupe: role+turnId+text key
                    const key = `${item.role}|${item.turnId || 'no-turn'}|${item.text}`;
                    if (seenMessageKeysRef.current.has(key)) {
                        console.log('⏭️ Skipping duplicate (global key match)');
                        return;
                    }

                    // Check if this is an update to an existing message (streaming)
                    setTranscript(prev => {
                        const lastItem = prev[prev.length - 1];
                        
                        // Check for EXACT duplicate (same role, text, turnId)
                        // This prevents double-adds from React strict mode or double renders
                        if (lastItem && 
                            lastItem.role === item.role &&
                            lastItem.text === item.text &&
                            lastItem.turnId === item.turnId) {
                            console.log(`⏭️ Skipping duplicate message`);
                            return prev; // Skip duplicate
                        }
                        
                        // Update existing message if:
                        // 1. Last item exists
                        // 2. Same turnId (if available) 
                        // 3. Same role
                        // 4. Different text (actual update)
                        const shouldUpdate = lastItem && 
                            lastItem.role === item.role &&
                            lastItem.text !== item.text &&
                            item.turnId && 
                            lastItem.turnId === item.turnId;
                        
                        if (shouldUpdate) {
                            // Update existing message (streaming continuation)
                            console.log(`🔄 Updating existing message at index ${prev.length - 1}`);
                            const updated = [...prev];
                            updated[updated.length - 1] = item;
                            return updated;
                        } else {
                            // New message - add to transcript
                            console.log(`➕ Adding new message (role: ${item.role}, turnId: ${item.turnId})`);
                            const added = [...prev, item];
                            seenMessageKeysRef.current.add(key);
                            return added;
                        }
                    });
                } else if (data.type === 'agent_state') {
                    const s = String(data.state || '').toUpperCase();
                    console.log('🤖 Agent state:', s);
                    // @ts-ignore - we trust backend states
                    setAgentState(s);
                } else if (data.type === 'speaking_state') {
                    // NEW: Explicit speaking lifecycle events from backend
                    const speaker = String(data.speaker || '');
                    const state = String(data.state || '');
                    console.log(`🔊 Speaking State: ${speaker} ${state}`);
                    
                    if (speaker === 'ai') {
                        if (state === 'start') {
                            setAiSpeaking(true);
                        } else if (state === 'end') {
                            setAiSpeaking(false);
                            const tId = typeof data.turnId === 'string' ? data.turnId : undefined;
                            const wasCanceled = Boolean(data.canceled);
                            if (tId) {
                                // Remove active streaming marker
                                setAiPlaybackStartedTurns(prev => {
                                    const next = new Set(prev);
                                    next.delete(tId);
                                    return next;
                                });
                                // Mark canceled turns so UI can freeze text where speech stopped
                                setAiPlaybackCanceledTurns(prev => {
                                    const next = new Set(prev);
                                    if (wasCanceled) {
                                        next.add(tId);
                                    } else {
                                        next.delete(tId);
                                    }
                                    return next;
                                });
                            }
                        }
                    } else if (speaker === 'user') {
                        if (state === 'start') {
                            setUserSpeaking(true);
                        } else if (state === 'end') {
                            setUserSpeaking(false);
                        }
                    }
                }
            } catch (err) {
                console.error('Error handling WS message:', err);
            }
        };

        websocket.onerror = (event) => {
            console.error('WebSocket error:', event);
            if (isMounted.current) setConnectionStatus('failed');
        };

        websocket.onclose = () => {
            console.log('⚠️ WebSocket closed - do NOT close peer connection (keep audio playing)');
            // DO NOT call cleanup here! Just log it.
            // The peer connection should stay alive to continue playing audio.
            // Only close it if the component unmounts via the useEffect cleanup.
            if (isMounted.current && peerConnection.connectionState === 'connected') {
                console.log('PC still connected, keeping it alive for audio playback');
            }
        };

        peerConnection.ontrack = (event) => {
            console.log('🎥 Remote Track received:', event.track.kind);
            setRemoteStream((prev) => {
                const stream = prev || new MediaStream();
                stream.addTrack(event.track);
                console.log(`📡 Added ${event.track.kind} track to remote stream`);
                return stream;
            });
        };

        peerConnection.onicecandidate = (event) => {
            if (event.candidate && websocket.readyState === WebSocket.OPEN) {
                console.log('❄️ Sending ICE candidate');
                websocket.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate.candidate,
                    sdpMid: event.candidate.sdpMid,
                    sdpMLineIndex: event.candidate.sdpMLineIndex
                }));
            }
        };

        peerConnection.onconnectionstatechange = () => {
            if (isMounted.current) {
                const newState = peerConnection.connectionState;
                console.log(`✅ [EVENT] Connection State Changed: ${newState}`);
                setConnectionStatus(newState);
                if (newState === 'connected') {
                    console.log('🎉 PEER CONNECTION IS NOW CONNECTED!');
                }
            }
        };

        peerConnection.oniceconnectionstatechange = () => {
            const iceState = peerConnection.iceConnectionState;
            console.log(`🧊 [EVENT] ICE Connection State: ${iceState}`);
            if (isMounted.current) {
                if (iceState === 'connected' || iceState === 'completed') {
                    console.log('🎉 ICE CONNECTED - FORCING CONNECTION STATUS TO CONNECTED');
                    setConnectionStatus('connected');
                } else if (iceState === 'disconnected' || iceState === 'failed') {
                    console.log('❌ ICE FAILED - SETTING STATUS TO FAILED');
                    setConnectionStatus('failed');
                }
            }
        };

    }, [roomId]);

    useEffect(() => {
        isMounted.current = true;
        let stateCheckInterval: NodeJS.Timeout | null = null;

        const setupConnection = async () => {
            // Only initialize if no peer connection exists or if it's in a bad state
            if (pc.current && pc.current.connectionState !== 'closed' && pc.current.connectionState !== 'failed') {
                console.log(`Peer connection already exists in state: ${pc.current.connectionState}, skipping init`);
                return;
            }
            
            await initialize();
        };

        setupConnection();

        // Poll connection state periodically as fallback
        stateCheckInterval = setInterval(() => {
            if (!isMounted.current || !pc.current) {
                if (stateCheckInterval) clearInterval(stateCheckInterval);
                return;
            }
            const currentState = pc.current.connectionState;
            const currentICEState = pc.current.iceConnectionState;
            
            // If ICE is connected but we're not reporting connected, fix it
            if ((currentICEState === 'connected' || currentICEState === 'completed') && 
                connectionStatus !== 'connected' && currentState !== 'connected') {
                console.log(`📍 State poll: ICE=${currentICEState}, PC=${currentState} -> forcing connected`);
                setConnectionStatus('connected');
            }
        }, 500);

        return () => {
            isMounted.current = false;
            if (stateCheckInterval) clearInterval(stateCheckInterval);
            cleanup();
        };
    }, [roomId]); // Only depend on roomId

    const reconnect = useCallback(() => {
        cleanup();
        setTimeout(() => {
            if (isMounted.current) {
                setRemoteStream(null);
                initialize();
            }
        }, 500);
    }, [cleanup, initialize]);


    // Hardware Privacy Logic: Stop tracks completely
    const toggleMute = useCallback(async () => {
        if (isMuted) {
            // Unmute: Re-acquire audio
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                const newTrack = stream.getAudioTracks()[0];

                if (localStreamRef.current) {
                    // Remove ALL old audio tracks first
                    localStreamRef.current.getAudioTracks().forEach(track => {
                        localStreamRef.current?.removeTrack(track);
                    });
                    // Add new track
                    localStreamRef.current.addTrack(newTrack);
                    setLocalStream(new MediaStream(localStreamRef.current.getTracks()));
                }

                if (audioSenderRef.current) {
                    await audioSenderRef.current.replaceTrack(newTrack);
                }
                setIsMuted(false);
            } catch (e) {
                console.error("Failed to unmute (acquire audio):", e);
            }
        } else {
            // Mute: Stop audio track
            if (localStreamRef.current) {
                localStreamRef.current.getAudioTracks().forEach(track => {
                    track.stop();
                    localStreamRef.current?.removeTrack(track);
                });
                setLocalStream(new MediaStream(localStreamRef.current.getTracks()));
            }
            if (audioSenderRef.current) {
                // Optionally replace with null or just let it handle the stopped track
                // audioSenderRef.current.replaceTrack(null); 
            }
            setIsMuted(true);
        }
    }, [isMuted]);

    const toggleVideo = useCallback(async () => {
        if (isVideoOff) {
            // Turn Video On: Re-acquire
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                const newTrack = stream.getVideoTracks()[0];

                if (localStreamRef.current) {
                    localStreamRef.current.addTrack(newTrack);
                    setLocalStream(new MediaStream(localStreamRef.current.getTracks()));
                }

                if (videoSenderRef.current) {
                    videoSenderRef.current.replaceTrack(newTrack).catch(e => console.error("Replace Video Error", e));
                }
                setIsVideoOff(false);
            } catch (e) {
                console.error("Failed to turn on video:", e);
            }
        } else {
            // Turn Video Off: Stop track
            if (localStreamRef.current) {
                localStreamRef.current.getVideoTracks().forEach(track => {
                    track.stop();
                    localStreamRef.current?.removeTrack(track); // Clean removal
                });
                // Force state update for UI to know track is gone (optional, isVideoOff handles UI)
                setLocalStream(new MediaStream(localStreamRef.current.getTracks()));
            }
            setIsVideoOff(true);
        }
    }, [isVideoOff]);

    const cancelTTS = useCallback(() => {
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(JSON.stringify({ type: 'cancel_tts' }));
        }
    }, []);

    return {
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
        userSpeaking, // NEW: Export user speaking state
        setAiSpeaking,
        transcript,
        aiPlaybackStartedTurns,
        aiPlaybackCanceledTurns,
        agentState,
        cancelTTS
    };
};
