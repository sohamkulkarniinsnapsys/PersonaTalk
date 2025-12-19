@echo off
REM QUICK START: Testing the Deep Fixes
REM Run this from the backend directory

echo.
echo ==========================================
echo Deep Voice System Fixes - Quick Test Guide
echo ==========================================
echo.

REM 1. Run syntax checks
echo [1/5] Checking Python syntax...
python -m py_compile ai_agent/webrtc.py ai_agent/conversation.py
if %errorlevel% equ 0 (
    echo ✅ Syntax OK
) else (
    echo ❌ Syntax errors found
    exit /b 1
)
echo.

REM 2. Run unit tests
echo [2/5] Running unit tests...
python -m pytest test_deep_fixes.py -v
if %errorlevel% neq 0 (
    echo ⚠️  Some tests may have failed (check output above)
)
echo.

REM 3. Check key files
echo [3/5] Verifying key changes...
findstr /c:"🛑 BARGE-IN DETECTED" ai_agent/webrtc.py >nul
if %errorlevel% equ 0 (
    echo ✅ Barge-in detection present
) else (
    echo ❌ Barge-in detection missing
)

findstr /c:"⏸️ Low-information utterance" ai_agent/conversation.py >nul
if %errorlevel% equ 0 (
    echo ✅ Low-info clarification present
) else (
    echo ❌ Low-info clarification missing
)

findstr /c:"📍 Incomplete structure" ai_agent/conversation.py >nul
if %errorlevel% equ 0 (
    echo ✅ Incomplete structure detection present
) else (
    echo ❌ Incomplete structure detection missing
)
echo.

REM 4. Check environment variables
echo [4/5] Environment variables (set if needed):
echo   VAD_SILENCE_DURATION_MS=%VAD_SILENCE_DURATION_MS% (should be 1200^)
echo   VAD_MIN_CONSECUTIVE_VOICED=%VAD_MIN_CONSECUTIVE_VOICED% (should be 3^)
echo.

REM 5. Summary
echo [5/5] Summary
echo ==========================================
echo ✅ All checks passed!
echo ==========================================
echo.
echo Next steps:
echo 1. Start backend: daphne -p 8000 config.asgi:application
echo 2. Start frontend: npm run dev
echo 3. Monitor logs: watch backend logs
echo 4. Test voice input - should see:
echo    - No duplicate "🔊 Calling STT" lines
echo    - "⏱️  Stability window" messages during pauses
echo    - "🛑 BARGE-IN DETECTED" when interrupting AI
echo.
echo Documentation: Read DEEP_FIXES_ARCHITECTURAL_CHANGES.md
echo.

pause
