const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

/** ---------- Chat (agent) ---------- */
export async function chat(message: string) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error("chat failed");
  return res.json();
}

/** ---------- Text search (server RAG) ----------
 * Flexible: { q|query|message, filters: {category?, color?}, min/max via NL like "under 75"
 */
export async function textSearch(q: string, k: number = 12, filters?: { category?: string; color?: string }) {
  const payload: any = { q, k };
  if (filters?.category || filters?.color) payload.filters = filters;
  const res = await fetch(`${API_BASE}/api/search_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error("text search failed");
  return res.json();
}

/** ---------- Image search by file (multipart) ---------- */
export async function imageSearchFile(file: File, k: number = 8) {
  const form = new FormData();
  form.append("file", file);
  form.append("k", String(k));
  const res = await fetch(`${API_BASE}/api/search_image`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("image search failed");
  return res.json();
}

/** ---------- Image search by URL ---------- */
export async function imageSearchUrl(url: string, k: number = 8) {
  const res = await fetch(`${API_BASE}/api/search_by_url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, k }),
  });
  if (!res.ok) throw new Error("image url search failed");
  return res.json();
}

/** ---------- Catalog ---------- */
export async function catalog() {
  const res = await fetch(`${API_BASE}/api/catalog`);
  if (!res.ok) throw new Error("catalog failed");
  return res.json();
}
