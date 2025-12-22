#!/usr/bin/env python3
"""
Quick verification script to test if voice capture fix is working.
Run this after restarting the backend to verify thresholds are correct.

Usage:
    python verify_voice_fix.py
"""

import subprocess
import sys
import time
import re

def check_backend_logs():
    """Check if backend is running and has the new thresholds."""
    print("=" * 70)
    print("🔍 VOICE CAPTURE FIX VERIFICATION")
    print("=" * 70)
    print()
    
    # Try to find daphne process and check its logs
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True, shell=True)
        if 'python' in result.stdout or 'daphne' in result.stdout:
            print("✅ Backend process appears to be running")
        else:
            print("⚠️  Could not confirm backend is running")
            print("   Make sure to run: cd backend && daphne -p 8000 config.asgi:application")
            return False
    except Exception as e:
        print(f"⚠️  Could not check process status: {e}")
    
    print()
    print("After restarting backend, you should see these log lines:")
    print()
    print("Expected Log Entry 1:")
    print("  VAD settings: start>=500, continue>=250, silence=2500ms, min_buffer=1500ms, ...")
    print()
    print("Expected Log Entry 2:")
    print("  VAD sensitivity: START_SNR_DB=6dB, MIN_CONSECUTIVE_START=3 frames (60ms), ")
    print("  Grace period=1200ms with 1.3x multiplier")
    print()
    print("If you see '3.0x multiplier' - the fix was NOT applied properly!")
    print()
    
    print("=" * 70)
    print("🧪 TEST SCENARIOS")
    print("=" * 70)
    print()
    
    tests = [
        {
            "name": "Test 1: Normal Speaking",
            "instruction": "Speak at normal volume (not loud, not quiet)",
            "expected": "All speech captured in single utterance",
            "success_log": "🎤 Speech START detected",
        },
        {
            "name": "Test 2: Loud Speaking",
            "instruction": "Speak loudly (at full volume mic)",
            "expected": "All loud speech captured",
            "success_log": "✅ Utterance complete: speech+silence window hit",
        },
        {
            "name": "Test 3: Natural Pauses",
            "instruction": "Speak, pause naturally 1 sec, continue",
            "expected": "Entire utterance captured as single turn",
            "success_log": "buffer=XXXX bytes (should be large, not small)",
        },
        {
            "name": "Test 4: Grace Period",
            "instruction": "Let AI finish response, then speak during grace (0-1.2s)",
            "expected": "Your speech captured within grace period",
            "success_log": "🎤 Speech START detected (during grace)",
        },
    ]
    
    for i, test in enumerate(tests, 1):
        print(f"{test['name']}")
        print(f"  Instruction: {test['instruction']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Success indicator in logs: '{test['success_log']}'")
        print()
    
    print("=" * 70)
    print("✅ HOW TO VERIFY IN LOGS")
    print("=" * 70)
    print()
    print("Watch the backend console (where you ran daphne):")
    print()
    print("Good signs (fix is working):")
    print("  ✅ 🎤 Speech START detected (peak=8500, snr=8.2 dB, dyn_start=650)")
    print("  ✅ ✅ Utterance complete: speech+silence window hit (buffer=98304 bytes)")
    print("  ✅ Buffer size should be > 10KB for normal speech")
    print()
    print("Bad signs (fix not working):")
    print("  ❌ 🛑 Speech start threshold error (dyn_start=1500 means 3.0x still active!)")
    print("  ❌ ⏳ Buffer too small at finalize (would mean speech cut off)")
    print("  ❌ ⚠️  Sarvam AI returned empty transcript (audio not reaching STT)")
    print()
    
    print("=" * 70)
    print("📝 CHANGES APPLIED")
    print("=" * 70)
    print()
    print("1. Grace period multiplier: 3.0x → 1.3x")
    print("2. Normal multiplier: 8x → 5x (start), 5x → 3x (continue)")
    print("3. START_SNR_DB: 10dB → 6dB")
    print("4. MIN_CONSECUTIVE_START: 6 frames → 3 frames (120ms → 60ms)")
    print("5. Silence SNR threshold: 4.0dB → 2.0dB")
    print()
    
    print("=" * 70)
    print("🚀 QUICK START")
    print("=" * 70)
    print()
    print("Step 1: Restart backend")
    print("  cd backend")
    print("  daphne -p 8000 config.asgi:application")
    print()
    print("Step 2: Watch logs for:")
    print("  'VAD sensitivity: START_SNR_DB=6dB'")
    print("  'Grace period=1200ms with 1.3x multiplier'")
    print()
    print("Step 3: Test in browser at http://localhost:3000/dashboard")
    print("  Speak at normal volume - should be captured")
    print()
    print("✅ Voice capture fix is working if you see the new threshold values in logs!")
    print()
    
    return True

if __name__ == '__main__':
    check_backend_logs()
