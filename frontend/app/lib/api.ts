export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function apiFetch(path: string, init: RequestInit = {}) {
  const url = path.startsWith('http') ? path : `${API_URL}${path}`;
  return fetch(url, {
    ...init,
    credentials: init.credentials ?? 'include',
  });
}
