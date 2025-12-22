'use client';

import React from 'react';

interface AIAnimationProps {
    isSpeaking: boolean;
    userSpeaking: boolean;
    personaName: string;
}

export const AIAnimation: React.FC<AIAnimationProps> = ({ isSpeaking, userSpeaking, personaName }) => {
    const isAnySpeaking = isSpeaking || userSpeaking;
    
    return (
        <div className="flex flex-col items-center justify-center space-y-6">
            {/* Animated avatar circle with color-coded effects */}
            <div className="relative w-40 h-40">
                {/* Outer glow aura when AI speaks (GREEN) */}
                {isSpeaking && (
                    <>
                        <div className="absolute -inset-2 rounded-full bg-green-400 opacity-40 blur-xl animate-pulse"></div>
                        <div className="absolute -inset-4 rounded-full bg-green-300 opacity-25 blur-2xl animate-pulse" 
                             style={{ animationDuration: '1.5s', animationDelay: '0.3s' }}></div>
                        <div className="absolute -inset-6 rounded-full bg-green-200 opacity-15 blur-3xl animate-pulse" 
                             style={{ animationDuration: '2s', animationDelay: '0.6s' }}></div>
                    </>
                )}
                
                {/* Outer glow aura when user speaks (BLUE) */}
                {userSpeaking && !isSpeaking && (
                    <>
                        <div className="absolute -inset-2 rounded-full bg-blue-400 opacity-40 blur-xl animate-pulse"></div>
                        <div className="absolute -inset-4 rounded-full bg-blue-300 opacity-25 blur-2xl animate-pulse" 
                             style={{ animationDuration: '1.5s', animationDelay: '0.3s' }}></div>
                        <div className="absolute -inset-6 rounded-full bg-blue-200 opacity-15 blur-3xl animate-pulse" 
                             style={{ animationDuration: '2s', animationDelay: '0.6s' }}></div>
                    </>
                )}
                
                {/* Main avatar circle with dynamic color changes */}
                <div className={`absolute inset-0 rounded-full flex items-center justify-center transition-all duration-300 ${
                    isSpeaking 
                        ? 'bg-gradient-to-br from-green-500 to-green-600 border-4 border-green-300 shadow-[0_0_50px_rgba(74,222,128,0.8)] scale-110' 
                        : userSpeaking
                        ? 'bg-gradient-to-br from-blue-500 to-blue-600 border-4 border-blue-300 shadow-[0_0_50px_rgba(59,130,246,0.8)] scale-110'
                        : 'bg-gradient-to-br from-indigo-600 to-purple-700 border-4 border-gray-700 shadow-lg scale-100'
                }`}>
                    {/* Microphone icon - always visible, pulses when speaking */}
                    <div className={`transition-all duration-300 ${
                        isAnySpeaking ? 'animate-pulse' : ''
                    }`}>
                        <svg className="w-20 h-20 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                        </svg>
                    </div>
                </div>
            </div>

            {/* Status text with color-coded states */}
            <div className="text-center">
                <h2 className="text-2xl font-bold text-white">{personaName}</h2>
                <p className={`text-sm mt-2 transition-all duration-300 font-semibold ${
                    isSpeaking 
                        ? 'text-green-400 scale-105' 
                        : userSpeaking
                        ? 'text-blue-400 scale-105'
                        : 'text-gray-400 scale-100'
                }`}>
                    {isSpeaking ? '🎤 AI Speaking' : userSpeaking ? '🎤 You are Speaking' : '👂 Listening'}
                </p>
            </div>
        </div>
    );
};
