#!/usr/bin/env python3
"""
Quick verification that voice capture fix is applied.
Run this after restarting backend to confirm all changes are in place.
"""

import subprocess
import re
import sys
import time

def check_webrtc_file():
    """Check if webrtc.py has the new thresholds."""
    print("\n" + "="*70)
    print("🔍 CHECKING VOICE CAPTURE FIX STATUS")
    print("="*70)
    
    try:
        with open('backend/ai_agent/webrtc.py', 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print("❌ ERROR: webrtc.py not found. Are you in the right directory?")
        print("   Should be in: c:\\insnapsys\\video_conf")
        return False
    
    checks = {
        'Grace Multiplier 1.3': 'grace_multiplier = 1.3',
        'START_SNR_DB = 6': 'START_SNR_DB = float(os.environ.get("VAD_START_SNR_DB", "6"))',
        'MIN_CONSECUTIVE_START = 3': 'MIN_CONSECUTIVE_START = int(os.environ.get("VAD_MIN_CONSECUTIVE_START", "3"))',
        'Grace Noise 10x': 'noise_floor * 32767 * 10',  # Grace period start
        'Normal Start 5x': 'noise_floor * 32767 * 5)  # REDUCED from 8x',
        'Silence SNR 2.0': 'snr_db < 2.0  # REDUCED from 4.0',
        'Enhanced Logging': 'VAD sensitivity:',
    }
    
    all_good = True
    for check_name, pattern in checks.items():
        if pattern in content:
            print(f"✅ {check_name:30s} FOUND")
        else:
            print(f"❌ {check_name:30s} NOT FOUND - FIX NOT APPLIED!")
            all_good = False
    
    return all_good

def check_backend_running():
    """Check if backend is running."""
    print("\n" + "="*70)
    print("🚀 CHECKING BACKEND STATUS")
    print("="*70)
    
    try:
        result = subprocess.run(
            ['netstat', '-ano'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if ':8000' in result.stdout:
            print("✅ Backend is running on port 8000")
            return True
        else:
            print("❌ Backend is NOT running on port 8000")
            print("\nTo start backend:")
            print("  cd backend")
            print("  daphne -p 8000 config.asgi:application")
            return False
    except Exception as e:
        print(f"⚠️  Could not check port status: {e}")
        return None

def show_expected_logs():
    """Show what logs should look like."""
    print("\n" + "="*70)
    print("📊 EXPECTED LOG OUTPUT (after backend restart)")
    print("="*70)
    
    print("""
Look for these in backend console:

✅ GOOD (Fix is working):
   VAD settings: start>=500, continue>=250, silence=2500ms, ...
   VAD sensitivity: START_SNR_DB=6dB, MIN_CONSECUTIVE_START=3 frames (60ms),
   Grace period=1200ms with 1.3x multiplier
   
   🎤 Speech START detected (peak=9234, snr=9.2 dB, dyn_start=650)
   ✅ Utterance complete: speech+silence window hit (buffer=98304 bytes)

❌ BAD (Fix NOT applied):
   Grace period=1200ms with 3.0x multiplier  ← OLD VALUE!
   dyn_start=1500  ← TOO HIGH!
""")

def test_scenarios():
    """Describe test scenarios."""
    print("\n" + "="*70)
    print("🧪 VERIFICATION TEST SCENARIOS")
    print("="*70)
    
    tests = [
        ("Test 1: Normal Speaking", 
         "Speak at normal volume → Should capture all speech",
         "✅ See 'Speech START detected' in logs"),
        
        ("Test 2: Loud Speaking (Critical!)", 
         "Speak very loudly → Should capture everything (this was failing before)",
         "✅ See full utterance in logs with large buffer"),
        
        ("Test 3: Natural Pauses", 
         "Speak, pause 1 second, continue → Should be ONE utterance",
         "✅ See 'Utterance complete' with buffer size > 10KB"),
        
        ("Test 4: Grace Period", 
         "Let AI finish, speak during grace (0-1.2s after) → Should capture",
         "✅ Speech detected without 3x multiplier blocking"),
    ]
    
    for test_num, action, expected in tests:
        print(f"\n{test_num}")
        print(f"  Action: {action}")
        print(f"  Expected: {expected}")

def main():
    print("\n")
    print("╔" + "═"*68 + "╗")
    print("║" + "  VOICE CAPTURE FIX VERIFICATION TOOL                        ".center(68) + "║")
    print("║" + "  Check if all fixes are applied and working                ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    
    # Check 1: File changes
    file_ok = check_webrtc_file()
    
    # Check 2: Backend running
    backend_ok = check_backend_running()
    
    # Show expected logs
    show_expected_logs()
    
    # Show test scenarios
    test_scenarios()
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY")
    print("="*70)
    
    if file_ok:
        print("✅ All code changes are in place")
    else:
        print("❌ Code changes NOT found - restart backend after applying changes!")
    
    if backend_ok:
        print("✅ Backend is running - ready to test voice capture")
    elif backend_ok is False:
        print("❌ Backend not running - start it to test")
    else:
        print("⚠️  Could not verify backend status")
    
    print("\n" + "="*70)
    print("🎯 NEXT STEPS")
    print("="*70)
    
    if not file_ok:
        print("""
1. Make sure you're in: c:\\insnapsys\\video_conf
2. Check that webrtc.py has all 5 changes applied
3. If not, apply the changes from the documentation
""")
    
    if backend_ok is False:
        print("""
1. Open a PowerShell terminal
2. Run: cd c:\\insnapsys\\video_conf\\backend
3. Run: daphne -p 8000 config.asgi:application
4. Wait for "Starting server..." message
5. Re-run this verification script
""")
    
    if file_ok and backend_ok:
        print("""
✅ EVERYTHING LOOKS GOOD!

1. Open browser: http://localhost:3000/dashboard
2. Start a call
3. Speak at normal volume
4. ✅ Expected: Speech captured immediately

Troubleshooting:
- Check backend console for "VAD sensitivity: START_SNR_DB=6dB... 1.3x multiplier"
- If you see "3.0x multiplier", the fix was not applied
- Check logs during speech for "Speech START detected"
""")
    
    print("\n" + "="*70)
    print("📞 NEED HELP?")
    print("="*70)
    print("""
Detailed Guides:
  - VOICE_CAPTURE_DEPLOYMENT_GUIDE.md (step-by-step)
  - VOICE_CAPTURE_FIX_FINAL_2025-12-18.md (technical details)
  - DEEP_FIXES_ARCHITECTURAL_CHANGES.md (architecture overview)
""")
    print("="*70 + "\n")
    
    return file_ok and backend_ok

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
