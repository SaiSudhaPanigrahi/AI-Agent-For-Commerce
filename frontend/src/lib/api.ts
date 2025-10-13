const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export async function chat(message: string) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'demo', message })
  });
  return res.json();
}

export async function recommend(query: string) {
  const res = await fetch(`${API_BASE}/api/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'demo', query })
  });
  return res.json();
}

export async function imageSearch(image_url: string) {
  const form = new FormData();
  form.append('image_url', image_url);
  const res = await fetch(`${API_BASE}/api/image-search`, {
    method: 'POST',
    body: form
  });
  return res.json();
}

export async function catalog() {
  const res = await fetch(`${API_BASE}/api/catalog`);
  return res.json();
}
