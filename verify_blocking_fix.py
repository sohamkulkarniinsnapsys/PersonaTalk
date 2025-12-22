#!/usr/bin/env python3
"""
Verification script for Blocking Audio Fix
Validates that all 9 await self.send_audio() calls have been converted to non-blocking
"""

import os
import sys
import re

def verify_blocking_audio_fix():
    """Verify all blocking audio calls have been converted to non-blocking."""
    
    print("=" * 80)
    print("🔍 BLOCKING AUDIO FIX VERIFICATION")
    print("=" * 80)
    print()
    
    # Check if file exists
    backend_path = os.path.join(os.path.dirname(__file__), 'backend')
    conversation_file = os.path.join(backend_path, 'ai_agent', 'conversation.py')
    
    if not os.path.exists(conversation_file):
        print(f"❌ File not found: {conversation_file}")
        return False
    
    print(f"📄 Checking: {conversation_file}")
    print()
    
    # Read file
    with open(conversation_file, 'r') as f:
        content = f.read()
    
    # Check 1: Count blocking calls (should be 0 or very few)
    blocking_pattern = r'await\s+self\.send_audio\s*\('
    blocking_matches = re.finditer(blocking_pattern, content)
    blocking_count = sum(1 for _ in blocking_matches)
    
    print("Check 1️⃣: Blocking `await self.send_audio()` calls")
    if blocking_count == 0:
        print("✅ PASS: No blocking calls found")
    else:
        # Re-find for reporting
        blocking_lines = []
        for i, line in enumerate(content.split('\n'), 1):
            if re.search(blocking_pattern, line):
                blocking_lines.append((i, line.strip()))
        
        print(f"❌ FAIL: Found {blocking_count} blocking calls:")
        for line_no, line_text in blocking_lines:
            print(f"   Line {line_no}: {line_text}")
    
    print()
    
    # Check 2: Count non-blocking calls (should be 9)
    nonblocking_pattern = r'asyncio\.ensure_future\(self\.send_audio\s*\('
    nonblocking_matches = list(re.finditer(nonblocking_pattern, content))
    nonblocking_count = len(nonblocking_matches)
    
    print("Check 2️⃣: Non-blocking `asyncio.ensure_future(self.send_audio)` calls")
    print(f"Found: {nonblocking_count} non-blocking calls")
    
    if nonblocking_count == 9:
        print("✅ PASS: All 9 expected non-blocking calls found")
        
        # Show locations
        print("\nLocations of non-blocking calls:")
        for i, match in enumerate(nonblocking_matches, 1):
            line_no = content[:match.start()].count('\n') + 1
            # Extract line context
            lines = content.split('\n')
            if 0 <= line_no - 1 < len(lines):
                print(f"   {i}. Line {line_no}: {lines[line_no-1].strip()[:70]}")
    elif nonblocking_count < 9:
        print(f"❌ FAIL: Expected 9 non-blocking calls, found {nonblocking_count}")
    else:
        print(f"⚠️  WARNING: Found {nonblocking_count} non-blocking calls (expected 9)")
    
    print()
    
    # Check 3: Verify no dangling await asyncio.sleep() after send_audio
    print("Check 3️⃣: Cleanup of unnecessary `await asyncio.sleep()` calls")
    
    # Look for patterns like:
    # await self.send_audio(...)
    # await asyncio.sleep(...)
    dangerous_pattern = r'await\s+self\.send_audio.*?\n.*?await\s+asyncio\.sleep'
    dangerous_matches = list(re.finditer(dangerous_pattern, content, re.DOTALL))
    
    if len(dangerous_matches) == 0:
        print("✅ PASS: No unnecessary sleep() calls after send_audio()")
    else:
        print(f"⚠️  WARNING: Found {len(dangerous_matches)} potential unnecessary sleeps")
        print("   (These may have been intended as sleeps, verify manually)")
    
    print()
    
    # Check 4: Verify asyncio import exists
    print("Check 4️⃣: Required import statements")
    
    has_asyncio = 'import asyncio' in content
    if has_asyncio:
        print("✅ PASS: `import asyncio` found")
    else:
        print("❌ FAIL: `import asyncio` not found")
    
    print()
    
    # Check 5: Syntax check
    print("Check 5️⃣: Python syntax validation")
    try:
        import py_compile
        py_compile.compile(conversation_file, doraise=True)
        print("✅ PASS: File has valid Python syntax")
    except py_compile.PyCompileError as e:
        print(f"❌ FAIL: Syntax error in file: {e}")
        return False
    
    print()
    print("=" * 80)
    
    # Final verdict
    all_pass = (blocking_count == 0 and nonblocking_count == 9 and has_asyncio)
    
    if all_pass:
        print("✅ ALL CHECKS PASSED")
        print()
        print("Status: Fix successfully applied and verified")
        print("Ready for: User testing of stop-word interruption")
        print()
        print("Next steps:")
        print("1. Restart backend: daphne -p 8000 config.asgi:application")
        print("2. Test stop-word: Say 'STOP' during AI speech")
        print("3. Expected: TTS cancels immediately")
        return True
    else:
        print("❌ SOME CHECKS FAILED")
        print()
        print("Issues found:")
        if blocking_count > 0:
            print(f"   • {blocking_count} blocking calls still present")
        if nonblocking_count != 9:
            print(f"   • Expected 9 non-blocking calls, found {nonblocking_count}")
        if not has_asyncio:
            print(f"   • Missing asyncio import")
        return False

if __name__ == '__main__':
    success = verify_blocking_audio_fix()
    sys.exit(0 if success else 1)
