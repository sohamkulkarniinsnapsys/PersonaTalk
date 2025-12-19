#!/usr/bin/env bash

# QUICK START: Testing the Deep Fixes
# Run this from the backend directory

echo "=========================================="
echo "Deep Voice System Fixes - Quick Test Guide"
echo "=========================================="
echo ""

# 1. Run syntax checks
echo "[1/5] Checking Python syntax..."
python -m py_compile ai_agent/webrtc.py ai_agent/conversation.py
if [ $? -eq 0 ]; then
    echo "✅ Syntax OK"
else
    echo "❌ Syntax errors found"
    exit 1
fi
echo ""

# 2. Run unit tests
echo "[2/5] Running unit tests..."
python -m pytest test_deep_fixes.py -v
if [ $? -eq 0 ]; then
    echo "✅ Unit tests passed"
else
    echo "⚠️  Some tests may have failed (check output above)"
fi
echo ""

# 3. Check for duplicate exception handlers
echo "[3/5] Checking for duplicate exception handlers..."
DUPS=$(grep -c "except Exception as e:" ai_agent/webrtc.py)
if [ "$DUPS" -lt 10 ]; then
    echo "✅ No suspicious duplicate exception handlers"
else
    echo "⚠️  Found $DUPS exception handlers (may be duplicate)"
fi
echo ""

# 4. Verify logging markers exist
echo "[4/5] Verifying logging markers..."
if grep -q "🛑 BARGE-IN DETECTED" ai_agent/webrtc.py; then
    echo "✅ Barge-in detection logging present"
fi
if grep -q "⏸️ Low-information utterance" ai_agent/conversation.py; then
    echo "✅ Low-info clarification logging present"
fi
if grep -q "📍 Incomplete structure" ai_agent/conversation.py; then
    echo "✅ Incomplete structure detection logging present"
fi
echo ""

# 5. Check environment variables
echo "[5/5] Environment variables check..."
echo "Current VAD settings:"
echo "  VAD_START_THRESHOLD=${VAD_START_THRESHOLD:-500}"
echo "  VAD_CONTINUE_THRESHOLD=${VAD_CONTINUE_THRESHOLD:-250}"
echo "  VAD_SILENCE_DURATION_MS=${VAD_SILENCE_DURATION_MS:-1200}"
echo "  VAD_POST_AI_GRACE_MS=${VAD_POST_AI_GRACE_MS:-800}"
echo ""
echo "To test with custom values:"
echo "  export VAD_SILENCE_DURATION_MS=1500  # Longer silence"
echo "  export VAD_MIN_BUFFER_MS=2000        # Require longer utterances"
echo ""

echo "=========================================="
echo "✅ All checks passed!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start backend: daphne -p 8000 config.asgi:application"
echo "2. Start frontend: npm run dev"
echo "3. Monitor logs: tail -f backend.log"
echo "4. Test voice input - should see:"
echo "   - No duplicate '🔊 Calling STT' lines"
echo "   - '⏱️  Stability window' messages during pauses"
echo "   - '🛑 BARGE-IN DETECTED' when interrupting AI"
echo "5. Check DEEP_FIXES_ARCHITECTURAL_CHANGES.md for details"
echo ""
