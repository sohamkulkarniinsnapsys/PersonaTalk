'use client';

import React from 'react';

interface AIAnimationProps {
    isSpeaking: boolean;
    personaName: string;
}

export const AIAnimation: React.FC<AIAnimationProps> = ({ isSpeaking, personaName }) => {
    return (
        <div className="flex flex-col items-center justify-center space-y-6">
            {/* Animated avatar circle with gradient */}
            <div className="relative w-40 h-40">
                {/* Background pulsing circles */}
                {isSpeaking && (
                    <>
                        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-indigo-500 to-purple-600 opacity-20 animate-pulse"></div>
                        <div className="absolute inset-4 rounded-full bg-gradient-to-r from-blue-500 to-indigo-600 opacity-30 animate-ping" 
                             style={{ animationDuration: '2s' }}></div>
                    </>
                )}
                
                {/* Main avatar circle */}
                <div className={`absolute inset-0 rounded-full bg-gradient-to-br from-indigo-600 to-purple-700 flex items-center justify-center border-4 transition-all duration-300 ${
                    isSpeaking ? 'border-indigo-300 shadow-2xl shadow-indigo-500/50' : 'border-gray-700'
                }`}>
                    {/* Sound wave visualization when speaking */}
                    {isSpeaking && (
                        <div className="flex items-center justify-center gap-1">
                            <div className="w-1 rounded-full bg-white opacity-80 animate-pulse" style={{ height: '20px', animationDelay: '0s' }}></div>
                            <div className="w-1 rounded-full bg-white opacity-80 animate-pulse" style={{ height: '32px', animationDelay: '0.2s' }}></div>
                            <div className="w-1 rounded-full bg-white opacity-80 animate-pulse" style={{ height: '24px', animationDelay: '0.4s' }}></div>
                        </div>
                    )}
                    
                    {/* Default icon when not speaking */}
                    {!isSpeaking && (
                        <svg className="w-20 h-20 text-white" fill="currentColor" viewBox="0 0 24 24">
                            <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                        </svg>
                    )}
                </div>
            </div>

            {/* Status text */}
            <div className="text-center">
                <h2 className="text-2xl font-bold text-white">{personaName}</h2>
                <p className={`text-sm mt-2 transition-colors ${
                    isSpeaking 
                        ? 'text-green-400 font-semibold' 
                        : 'text-gray-400'
                }`}>
                    {isSpeaking ? '🎤 Speaking' : 'Ready to listen'}
                </p>
            </div>
        </div>
    );
};
