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
                    {/* Sound wave animation */}
                    <div className="flex items-end justify-center gap-1 h-16">
                        <div 
                            className="w-2 bg-gradient-to-t from-blue-400 to-blue-600 rounded-full"
                            style={{ 
                                height: '24px',
                                animation: 'soundWave 0.6s ease-in-out infinite',
                                animationDelay: '0s'
                            }}
                        ></div>
                        <div 
                            className="w-2 bg-gradient-to-t from-blue-400 to-blue-600 rounded-full"
                            style={{ 
                                height: '32px',
                                animation: 'soundWave 0.6s ease-in-out infinite',
                                animationDelay: '0.1s'
                            }}
                        ></div>
                        <div 
                            className="w-2 bg-gradient-to-t from-blue-400 to-blue-600 rounded-full"
                            style={{ 
                                height: '40px',
                                animation: 'soundWave 0.6s ease-in-out infinite',
                                animationDelay: '0.2s'
                            }}
                        ></div>
                        <div 
                            className="w-2 bg-gradient-to-t from-blue-400 to-blue-600 rounded-full"
                            style={{ 
                                height: '32px',
                                animation: 'soundWave 0.6s ease-in-out infinite',
                                animationDelay: '0.3s'
                            }}
                        ></div>
                        <div 
                            className="w-2 bg-gradient-to-t from-blue-400 to-blue-600 rounded-full"
                            style={{ 
                                height: '24px',
                                animation: 'soundWave 0.6s ease-in-out infinite',
                                animationDelay: '0.4s'
                            }}
                        ></div>
                    </div>

                    {/* Status text */}
                    <p className="text-blue-400 font-semibold text-sm">Listening...</p>

                    <style>{`
                        @keyframes soundWave {
                            0%, 100% {
                                height: 16px;
                                opacity: 0.6;
                            }
                            50% {
                                height: 48px;
                                opacity: 1;
                            }
                        }
                    `}</style>
                </div>
            )}
        </div>
    );
};
