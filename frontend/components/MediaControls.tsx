import React, { useState } from 'react';
import { Mic, MicOff, Video, VideoOff, PhoneOff } from 'lucide-react';
import { useRouter } from 'next/navigation';

interface MediaControlsProps {
    isMuted: boolean;
    isVideoOff: boolean;
    onToggleMute: () => void;
    onToggleVideo: () => void;
    onEndCall: () => void;
}

export const MediaControls: React.FC<MediaControlsProps> = ({
    isMuted,
    isVideoOff,
    onToggleMute,
    onToggleVideo,
    onEndCall,
}) => {
    const router = useRouter();
    const [isEnding, setIsEnding] = useState(false);

    const handleEndCall = async () => {
        console.log('🛑 End call clicked - starting cleanup...');
        setIsEnding(true);
        try {
            // Call cleanup immediately
            console.log('📞 Calling cleanup function...');
            onEndCall();
            
            // Wait for cleanup to finish processing
            console.log('⏳ Waiting 200ms for cleanup to complete...');
            await new Promise(resolve => setTimeout(resolve, 200));
            
            console.log('✅ Cleanup complete, navigating to dashboard...');
        } catch (error) {
            console.error('❌ Error during cleanup:', error);
        } finally {
            router.push('/dashboard');
        }
    };

    return (
        <div className="flex items-center justify-center gap-6 p-4 bg-gray-900/80 backdrop-blur-sm rounded-full shadow-lg">
            <button
                onClick={onToggleMute}
                disabled={isEnding}
                className={`cursor-pointer p-4 rounded-full transition-colors ${isMuted ? 'bg-red-500 hover:bg-red-600' : 'bg-gray-700 hover:bg-gray-600'
                    } ${isEnding ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={isMuted ? "Unmute" : "Mute"}
            >
                {isMuted ? <MicOff className="w-6 h-6 text-white" /> : <Mic className="w-6 h-6 text-white" />}
            </button>

            <button
                onClick={onToggleVideo}
                disabled={isEnding}
                className={`cursor-pointer p-4 rounded-full transition-colors ${isVideoOff ? 'bg-red-500 hover:bg-red-600' : 'bg-gray-700 hover:bg-gray-600'
                    } ${isEnding ? 'opacity-50 cursor-not-allowed' : ''}`}
                title={isVideoOff ? "Turn Video On" : "Turn Video Off"}
            >
                {isVideoOff ? <VideoOff className="w-6 h-6 text-white" /> : <Video className="w-6 h-6 text-white" />}
            </button>

            <button
                onClick={handleEndCall}
                disabled={isEnding}
                className={`cursor-pointer p-4 rounded-full bg-red-600 hover:bg-red-700 transition-colors ${isEnding ? 'opacity-50 cursor-not-allowed' : ''}`}
                title="End Call"
            >
                <PhoneOff className="w-6 h-6 text-white" />
            </button>
        </div>
    );
};
