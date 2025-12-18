# Authentication Redesign - Developer Guide

## Overview
The project now uses a custom-branded authentication system with a "Cosmic Depth" theme. The login page is the application root (`/`).

### Key Features
- **Frontend**: Tailwind CSS (CDN dev / Config provided for build), Vanilla JS interactions, Glassmorphism design.
- **Backend**: Standard Django `auth` views + `UserCreationForm`.
- **Security**: Validated redirects, CSRF protection, Password validators.

## Integration

### Files
- `templates/base_auth.html`: Main layout.
- `templates/registration/`: Login and Register forms.
- `static/css/theme.css`: Core variables and utilities.
- `static/js/auth.js`: Logic for animations and password strength.
- `api/views.py`: `RegisterView`.
- `config/urls.py`: Auth routing.

### Redirects
Default redirects are configured in `settings.py`:
```python
LOGIN_REDIRECT_URL = '/api/' # Change this to your main dashboard URL
LOGOUT_REDIRECT_URL = '/'
```

## Theming
The design relies on CSS variables in `static/css/theme.css`.

### Customizing Colors
Edit the `:root` block to change the palette:
```css
:root {
    --color-primary: #6366f1; /* Your Brand Color */
    --color-bg-deep: #0f172a; /* Background */
}
```

### Build Process (Optional)
Currently, Tailwind is loaded via CDN for simplicity. To switch to a production build:
1. Install tailwind: `npm install -D tailwindcss`
2. Run build: `npx tailwindcss -i ./static/src/input.css -o ./static/css/theme.css --watch`
3. Update `base_auth.html` to remove the CDN script.

## Testing & QA
### E2E Scenarios (Playwright Pseudocode)

```javascript
// tests/auth.spec.js

test('Login success redirect', async ({ page }) => {
    await page.goto('/');
    await page.fill('input[name="username"]', 'testuser');
    await page.fill('input[name="password"]', 'password');
    await page.click('button[type="submit"]');
    await expect(page).toHaveURL('/api/');
});

test('Register flow', async ({ page }) => {
    await page.goto('/accounts/register/');
    await page.fill('input[name="username"]', 'newuser');
    await page.fill('input[name="password"]', 'StrongPass1!');
    await page.fill('input[name="password2"]', 'StrongPass1!');
    await page.click('button[type="submit"]');
    // Expect redirect to login with success message
    await expect(page).toHaveURL('/');
    await expect(page.locator('.text-green-200')).toBeVisible();
});
```

### Security Notes
- **Brute Force**: Recommended to install `django-axes` for production.
- **Session**: Ensure `SESSION_COOKIE_SECURE = True` in production `settings.py`.
