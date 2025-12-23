import React, { useEffect, useRef, useState } from 'react';

export interface TranscriptItem {
  role: 'user' | 'assistant';
  text: string;
  timestamp?: number;
  utteranceId?: string;
  turnId?: string;
  isStreaming?: boolean; // Flag if this message is still streaming
  displayDelayMs?: number; // How long to delay before showing (for AI sync)
}

interface StreamingTextProps {
  text: string;
  isStreaming: boolean;
  role: 'user' | 'assistant';
  shouldFreezeOnStop?: boolean; // When true, do not auto-reveal remaining text when streaming stops
}

/**
 * Streaming text component that animates character-by-character.
 * Creates real-time generation effect with smooth, performant animation.
 */
function StreamingText({ text, isStreaming, role, shouldFreezeOnStop = false }: StreamingTextProps) {
  const [displayedText, setDisplayedText] = useState('');
  const [displayedCursorPos, setDisplayedCursorPos] = useState(0);
  const textLengthRef = useRef(text.length);

  useEffect(() => {
    textLengthRef.current = text.length;
  }, [text]);

  useEffect(() => {
    // If not streaming, decide whether to reveal or freeze at current cursor
    if (!isStreaming) {
      if (shouldFreezeOnStop) {
        const currentPos = Math.min(displayedCursorPos, text.length);
        setDisplayedCursorPos(currentPos);
        setDisplayedText(text.slice(0, currentPos));
        return;
      }
      setDisplayedText(text);
      setDisplayedCursorPos(text.length);
      return;
    }

    // If streaming, animate character by character
    // Start from current position to allow for incremental updates
    if (displayedCursorPos >= text.length) {
      setDisplayedText(text);
      return;
    }

    // Use requestAnimationFrame for smooth 60fps animation
    // Add characters at ~50 chars/sec (20ms per char = ~100 chars visible per frame)
    // Slow down to roughly match spoken pace (~12-15 chars/sec)
    const charDelay = 60; // milliseconds per character
    const timeout = setTimeout(() => {
      const newPos = displayedCursorPos + 1;
      setDisplayedCursorPos(newPos);
      setDisplayedText(text.slice(0, newPos));
    }, charDelay);

    return () => clearTimeout(timeout);
  }, [displayedCursorPos, text, isStreaming]);

  // When text content changes (new message or update), sync displayed text
  useEffect(() => {
    if (!isStreaming) {
      if (shouldFreezeOnStop) {
        const currentPos = Math.min(displayedCursorPos, text.length);
        setDisplayedText(text.slice(0, currentPos));
        setDisplayedCursorPos(currentPos);
      } else {
        setDisplayedText(text);
        setDisplayedCursorPos(text.length);
      }
    } else {
      // If message is new and streaming, start from beginning
      // This handles case where new messages arrive
      if (displayedCursorPos === 0 || displayedCursorPos > text.length) {
        setDisplayedCursorPos(0);
        setDisplayedText('');
      }
    }
  }, [text, isStreaming]);

  return (
    <span>
      {displayedText}
      {isStreaming && displayedCursorPos < text.length && (
        <span className="animate-pulse">▋</span>
      )}
    </span>
  );
}

interface MessageItemProps {
  item: TranscriptItem;
  isVisible: boolean;
  personaName?: string;
  playbackStartedTurns?: Set<string>;
  playbackCanceledTurns?: Set<string>;
}

/**
 * Individual message with delayed display capability.
 * Used to sync assistant messages with audio playback.
 */
function MessageItem({ item, isVisible, personaName, playbackStartedTurns, playbackCanceledTurns }: MessageItemProps) {
  if (!isVisible) {
    return null; // Hidden until delay expires
  }

  // Stream assistant text only while its turn is actively playing audio
  const isStreamingActive = item.role === 'assistant'
    ? Boolean(item.turnId && playbackStartedTurns?.has(item.turnId))
    : false;

  const isCanceled = item.role === 'assistant'
    ? Boolean(item.turnId && playbackCanceledTurns?.has(item.turnId))
    : false;

  return (
    <div
      className={`flex transition-all duration-300 ${item.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2`}
    >
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words transition-all duration-1300 ${
          item.role === 'user'
            ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg shadow-blue-500/20'
            : item.isStreaming
            ? 'bg-gray-700/80 text-gray-100 shadow-lg shadow-gray-700/20 ring-1 ring-gray-600/50'
            : 'bg-gray-700/60 text-gray-100 shadow-lg shadow-gray-700/10'
        }`}
        title={item.role === 'user' ? 'You' : personaName || 'AI'}
      >
        <StreamingText
          text={item.text}
          isStreaming={isStreamingActive || item.isStreaming || false}
          role={item.role}
          shouldFreezeOnStop={isCanceled}
        />
      </div>
    </div>
  );
}

export function TranscriptPanel({ 
  items, 
  personaName,
  aiSpeaking = false,
  playbackStartedTurns,
  playbackCanceledTurns
}: { 
  items: TranscriptItem[]; 
  personaName?: string;
  aiSpeaking?: boolean;
  playbackStartedTurns?: Set<string>;
  playbackCanceledTurns?: Set<string>;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [visibleItems, setVisibleItems] = useState<Set<number>>(new Set());

  // Detect user manual scroll - disable auto-scroll if they scroll up
  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
    setAutoScroll(isAtBottom);
  };

  // Handle display timing for messages
  useEffect(() => {
    items.forEach((item, idx) => {
      const isAlreadyVisible = visibleItems.has(idx);
      
      if (isAlreadyVisible) return; // Already showing, skip
      
      // For user messages, show immediately (user always hears themselves)
      if (item.role === 'user') {
        setVisibleItems(prev => new Set(prev).add(idx));
        return;
      }

      // For AI/assistant messages, gate visibility on playback start signal
      // Show when either:
      // 1) We received ai_playback start for this turnId
      // 2) Fallback timeout (max wait) to avoid never showing
      const checkAndReveal = () => {
        const started = !!(item.turnId && playbackStartedTurns && playbackStartedTurns.has(item.turnId));
        if (started) {
          setVisibleItems(prev => new Set(prev).add(idx));
          return true;
        }
        return false;
      };

      if (checkAndReveal()) return;

      const startTime = Date.now();
      const maxWaitMs = 15000; // 15s safety fallback (covers long TTS synthesis)
      const interval = setInterval(() => {
        if (checkAndReveal()) {
          clearInterval(interval);
        } else if (Date.now() - startTime > maxWaitMs) {
          // Fallback: reveal anyway after max wait
          setVisibleItems(prev => new Set(prev).add(idx));
          clearInterval(interval);
        }
      }, 100);

      return () => clearInterval(interval);
    });
  }, [items, visibleItems, playbackStartedTurns]);

  // Auto-scroll to bottom when new visible messages arrive
  useEffect(() => {
    if (autoScroll) {
      // Use a small delay to ensure DOM has updated
      const scrollTimer = setTimeout(() => {
        endRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 50);
      return () => clearTimeout(scrollTimer);
    }
  }, [visibleItems, autoScroll]);

  return (
    <div className="h-full w-full flex flex-col bg-gradient-to-b from-black/40 to-black/20">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700/50 text-sm font-semibold text-gray-300 backdrop-blur-sm flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full transition-colors ${aiSpeaking ? 'bg-red-500 animate-pulse' : 'bg-blue-500 animate-pulse'}`}></div>
          <span>Live Transcript</span>
        </div>
        {!autoScroll && (
          <button
            onClick={() => setAutoScroll(true)}
            className="text-xs px-2 py-1 bg-blue-600/50 hover:bg-blue-600 rounded transition-colors"
          >
            ↓ Latest
          </button>
        )}
      </div>

      {/* Messages Container */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent"
      >
        {items.length === 0 ? (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            <span className="animate-pulse">Waiting for first message...</span>
          </div>
        ) : (
          items.map((item, idx) => (
            <MessageItem
              key={`${item.turnId || item.utteranceId || idx}-${idx}`}
              item={item}
              isVisible={visibleItems.has(idx)}
              personaName={personaName}
              playbackStartedTurns={playbackStartedTurns}
              playbackCanceledTurns={playbackCanceledTurns}
            />
          ))
        )}
        <div ref={endRef} />
      </div>
    </div>
  );
}
