/**
 * Auth Page Interactions
 * Handles floating labels, switching between login/register (if valid),
 * and basic form validations.
 */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Entrance Animation Stagger
    const staggerElements = document.querySelectorAll('.stagger-entry');
    staggerElements.forEach((el, index) => {
        el.style.opacity = '0';
        el.style.animation = `fadeIn 0.5s ease forwards ${index * 0.1}s`;
    });

    // 2. Floating Labels Logic
    const inputs = document.querySelectorAll('.form-input');
    inputs.forEach(input => {
        // Init state
        updateLabel(input);

        input.addEventListener('focus', () => input.parentElement.classList.add('focused'));
        input.addEventListener('blur', () => {
            input.parentElement.classList.remove('focused');
            updateLabel(input);
        });
        input.addEventListener('input', () => updateLabel(input));
    });

    function updateLabel(input) {
        if (input.value && input.value.length > 0) {
            input.parentElement.classList.add('has-value');
        } else {
            input.parentElement.classList.remove('has-value');
        }
    }

    // 3. Password Toggle
    const toggleBtns = document.querySelectorAll('.password-toggle');
    toggleBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const input = btn.previousElementSibling;
            if (input.type === 'password') {
                input.type = 'text';
                btn.setAttribute('aria-label', 'Hide password');
                btn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" /></svg>';
            } else {
                input.type = 'password';
                btn.setAttribute('aria-label', 'Show password');
                btn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" /></svg>';
            }
        });
    });

    // 4. Reduced Motion Check
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (mediaQuery.matches) {
        document.querySelectorAll('.animate-entrance, .stagger-entry').forEach(el => {
            el.style.animation = 'none';
            el.style.opacity = '1';
            el.style.transform = 'none';
        });
    }

    // 5. Password Strength (Visual Only)
    const passwordInputs = document.querySelectorAll('input[type="password"]');
    passwordInputs.forEach(input => {
        if (!input.name.includes('password')) return;

        // Only add strength meter for registration (usually has multiple password fields or specific ID)
        // Heuristic: if there's a parent form with "register" in action or class, or just strictly check ID
        if (input.closest('form') && input.closest('form').action.includes('register')) {
            const meter = document.createElement('div');
            meter.className = 'h-1 mt-2 w-full bg-gray-700 rounded overflow-hidden transition-all';
            const bar = document.createElement('div');
            bar.className = 'h-full bg-red-500 width-0 transition-all duration-300';
            bar.style.width = '0%';
            meter.appendChild(bar);
            input.parentElement.appendChild(meter);

            input.addEventListener('input', () => {
                const val = input.value;
                let strength = 0;
                if (val.length > 5) strength += 1;
                if (val.length > 8) strength += 1;
                if (/[A-Z]/.test(val)) strength += 1;
                if (/[0-9]/.test(val)) strength += 1;
                if (/[^A-Za-z0-9]/.test(val)) strength += 1;

                if (val.length === 0) {
                    bar.style.width = '0%';
                } else if (strength < 2) {
                    bar.style.width = '20%';
                    bar.className = 'h-full bg-red-500 transition-all duration-300';
                } else if (strength < 4) {
                    bar.style.width = '60%';
                    bar.className = 'h-full bg-yellow-500 transition-all duration-300';
                } else {
                    bar.style.width = '100%';
                    bar.className = 'h-full bg-green-500 transition-all duration-300';
                }
            });
        }
    });

    // 6. CSRF for AJAX (Utility)
    window.getCookie = function (name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    };
});
