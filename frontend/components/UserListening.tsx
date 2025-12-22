'use client';

import React from 'react';

interface UserListeningProps {
    isListening: boolean;
}

export const UserListening: React.FC<UserListeningProps> = ({ isListening }) => {
    return (
        <div className="absolute bottom-24 left-1/2 transform -translate-x-1/2 z-40">
            {isListening && (
                <div className="flex flex-col items-center space-y-3">
                    {/* Mic activity ring - Google Meet style */}
                    <div className="relative">
                        <div className="absolute inset-0 rounded-full bg-red-500 opacity-30 blur-xl animate-pulse"></div>
                        <div className="relative w-16 h-16 rounded-full bg-gradient-to-br from-red-500 to-red-600 flex items-center justify-center shadow-[0_0_30px_rgba(239,68,68,0.6)] animate-pulse" 
                             style={{ animationDuration: '1.5s' }}>
                            <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                            </svg>
                        </div>
                    </div>

                    {/* Enhanced sound wave animation */}
                    <div className="flex items-end justify-center gap-1.5 h-16">
                        <div 
                            className="w-2.5 bg-gradient-to-t from-red-400 via-red-500 to-red-600 rounded-full shadow-lg transition-all duration-150"
                            style={{ 
                                height: '24px',
                                animation: 'soundWave 0.8s ease-in-out infinite',
                                animationDelay: '0s'
                            }}
                        ></div>
                        <div 
                            className="w-2.5 bg-gradient-to-t from-red-400 via-red-500 to-red-600 rounded-full shadow-lg transition-all duration-150"
                            style={{ 
                                height: '36px',
                                animation: 'soundWave 0.8s ease-in-out infinite',
                                animationDelay: '0.1s'
                            }}
                        ></div>
                        <div 
                            className="w-2.5 bg-gradient-to-t from-red-400 via-red-500 to-red-600 rounded-full shadow-lg transition-all duration-150"
                            style={{ 
                                height: '48px',
                                animation: 'soundWave 0.8s ease-in-out infinite',
                                animationDelay: '0.2s'
                            }}
                        ></div>
                        <div 
                            className="w-2.5 bg-gradient-to-t from-red-400 via-red-500 to-red-600 rounded-full shadow-lg transition-all duration-150"
                            style={{ 
                                height: '36px',
                                animation: 'soundWave 0.8s ease-in-out infinite',
                                animationDelay: '0.3s'
                            }}
                        ></div>
                        <div 
                            className="w-2.5 bg-gradient-to-t from-red-400 via-red-500 to-red-600 rounded-full shadow-lg transition-all duration-150"
                            style={{ 
                                height: '24px',
                                animation: 'soundWave 0.8s ease-in-out infinite',
                                animationDelay: '0.4s'
                            }}
                        ></div>
                    </div>

                    {/* Status text with glow effect */}
                    <p className="text-red-400 font-bold text-sm tracking-wide animate-pulse" 
                       style={{ textShadow: '0 0 10px rgba(248, 113, 113, 0.5)' }}>
                        🎙️ Speaking...
                    </p>

                    <style>{`
                        @keyframes soundWave {
                            0%, 100% {
                                height: 20px;
                                opacity: 0.7;
                            }
                            50% {
                                height: 56px;
                                opacity: 1;
                            }
                        }
                    `}</style>
                </div>
            )}
        </div>
    );
};
