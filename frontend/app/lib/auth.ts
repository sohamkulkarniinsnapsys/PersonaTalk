/**
 * Auth utilities for Django backend integration
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

/**
 * Get CSRF token from cookie
 */
export function getCookie(name: string): string | null {
    if (typeof document === 'undefined') return null;

    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === (name + '=')) {
            return decodeURIComponent(cookie.substring(name.length + 1));
        }
    }
    return null;
}

/**
 * Get CSRF token from Django
 */
export async function getCSRFToken(): Promise<string> {
    try {
        const response = await fetch(`${API_URL}/api/csrf/`, {
            credentials: 'include',
        });
        const data = await response.json();
        return data.csrfToken || getCookie('csrftoken') || '';
    } catch {
        return getCookie('csrftoken') || '';
    }
}

export async function login(username: string, password: string): Promise<{
    success: boolean;
    error?: string;
    errors?: { username?: string; password?: string };
    redirectUrl?: string;
}> {
    const csrfToken = await getCSRFToken();

    try {
        const response = await fetch(`${API_URL}/api/auth/login/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            credentials: 'include',
            body: JSON.stringify({ username, password }),
        });

        const data = await response.json();

        if (response.ok && data.success) {
            return {
                success: true,
                redirectUrl: data.redirectUrl || '/dashboard'
            };
        }

        // Return field-specific errors
        return {
            success: false,
            errors: data.errors || {},
            error: data.error || 'Login failed'
        };

    } catch (error) {
        console.error('Login error:', error);
        return {
            success: false,
            error: 'Network error. Please try again.'
        };
    }
}

/**
 * Register new user with Django backend
 */
export async function register(username: string, password1: string, password2: string): Promise<{
    success: boolean;
    error?: string;
    errors?: { username?: string; password?: string; password2?: string };
    redirectUrl?: string;
}> {
    const csrfToken = await getCSRFToken();

    try {
        const response = await fetch(`${API_URL}/api/auth/register/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken,
            },
            credentials: 'include',
            body: JSON.stringify({ username, password1, password2 }),
        });

        const data = await response.json();

        if (response.ok && data.success) {
            return {
                success: true,
                redirectUrl: data.redirectUrl || '/dashboard'
            };
        }

        // Return field-specific errors
        return {
            success: false,
            errors: data.errors || {},
            error: data.error || 'Registration failed'
        };

    } catch (error) {
        console.error('Registration error:', error);
        return {
            success: false,
            error: 'Network error. Please try again.'
        };
    }
}

/**
 * Check if user is authenticated
 */
export async function checkAuth(): Promise<boolean> {
    try {
        const response = await fetch(`${API_URL}/api/auth/check/`, {
            credentials: 'include',
        });
        return response.ok;
    } catch {
        return false;
    }
}

/**
 * Logout user
 */
export async function logout(): Promise<void> {
    const csrfToken = await getCSRFToken();

    try {
        await fetch(`/accounts/logout/`, {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken,
            },
            credentials: 'include',
        });
    } catch (error) {
        console.error('Logout error:', error);
    }
}
