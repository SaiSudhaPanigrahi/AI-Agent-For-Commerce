import React, { useEffect, useMemo, useState } from "react"
import logo from "./logo.svg";


type Item = {
  id: string
  title: string
  category: string
  color: string
  price: number
  description: string
  image_path: string
  score?: number
}

/** Lively but readable palette */
const BLUE = "#19E3FF"
const TEAL = "#19FFD2"
const BG0 = "#0E1430"                     // page background
const PANEL_BG = "rgba(20, 28, 60, 0.88)" // containers slightly lighter than BG
const CARD_BG = "rgba(16, 24, 50, 0.94)"  // cards a touch lighter than panel
const MUTED = "#A8C3D9"

const styles: Record<string, React.CSSProperties | any> = {
  page: {
    minHeight: "100%",
    background:
      `radial-gradient(1100px 540px at 18% -10%, rgba(25,227,255,.14), transparent 60%),
       radial-gradient(900px 420px at 85% -4%, rgba(25,255,210,.10), transparent 60%),
       linear-gradient(180deg, ${BG0} 0%, #0b1127 100%)`,
    color: "white",
    fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto",
  },

  // Layout tweaks
  wrap: {
    maxWidth: 1240,
    margin: "0 auto",
    padding: "48px 28px 112px",
    boxSizing: "border-box",
  },
  header: {
    display: "grid",
    gridTemplateColumns: "auto 1fr auto",
    alignItems: "center",
    gap: 20,
    marginBottom: 14,
  },
  logoBox: {
    width: 64,
    height: 64,
    borderRadius: 16,
    background: "linear-gradient(145deg, #0f1a3a, #0a1027)",
    boxShadow: "0 8px 24px rgba(0,0,0,.45), 0 0 0 1px rgba(255,255,255,.06) inset",
    display: "grid",
    placeItems: "center",
    overflow: "hidden",
  },
  logo: { width: 46, height: 46 },

  titleBlock: { display: "flex", flexDirection: "column" as const },
  title: { fontSize: 48, fontWeight: 900, letterSpacing: 0.3, lineHeight: 1.06 },
  ai: { color: BLUE, textShadow: `0 0 18px ${BLUE}66` },
  tagline: { color: MUTED, marginTop: 10, fontSize: 17, maxWidth: 1000 },

  docsBtn: {
    background: "transparent",
    border: `1px solid ${BLUE}`,
    color: BLUE,
    padding: "11px 16px",
    borderRadius: 12,
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: `0 0 16px ${BLUE}22`,
    textDecoration: "none",
  },

  /* --- Panels/Cards --- */
  panel: {
    background: PANEL_BG,
    backdropFilter: "blur(6px)",
    border: "none",
    borderRadius: 18,
    padding: 20,
    boxShadow: "0 14px 36px rgba(0,0,0,0.35), inset 0 0 0 1px rgba(255,255,255,0.05)",
    marginTop: 20,
  },
  inputRow: {
    display: "grid",
    gridTemplateColumns: "1fr auto auto",
    gap: 12,
    alignItems: "center",
  },
  textInput: {
    width: "100%",
    height: 50,
    background: "rgba(10,18,42,.8)",
    color: "#E8F4FF",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 12,
    padding: "0 14px",
    outline: "none",
    fontSize: 16,
  },
  primaryBtn: {
    background: BLUE,
    color: "#001018",
    border: "none",
    height: 50,
    padding: "0 18px",
    borderRadius: 12,
    fontWeight: 800,
    cursor: "pointer",
    boxShadow: `0 10px 28px ${BLUE}33, 0 0 0 1px ${BLUE}66 inset`,
  },
  ghostBtn: {
    background: "transparent",
    color: BLUE,
    border: `1px solid ${BLUE}`,
    height: 50,
    padding: "0 14px",
    borderRadius: 12,
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: `0 0 16px ${BLUE}22`,
  },
  hint: { color: MUTED, fontSize: 13.5, marginTop: 8 },

  twoCol: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 18,
  },
  textarea: {
    width: "100%",
    minHeight: 140,
    background: "rgba(10,18,42,.75)",
    color: "#E8F4FF",
    border: "1px solid rgba(255,255,255,0.06)",
    borderRadius: 14,
    padding: 14,
    whiteSpace: "pre-wrap",
    fontSize: 15,
  },

  grid: {
    marginTop: 14,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(270px, 1fr))",
    gap: 18,
  },
  card: {
    background: CARD_BG,
    border: "none",
    borderRadius: 16,
    padding: 14,
    boxShadow: "0 12px 28px rgba(0,0,0,.35)",
  },
  shot: {
    height: 220,
    background: "#0b1220",
    borderRadius: 12,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  price: { color: TEAL, fontWeight: 900, marginTop: 6, fontSize: 17 },
  meta: { color: "#D1E6F6", fontSize: 14, marginTop: 2, opacity: .9 },
  desc: { color: "#B4CBE0", fontSize: 14, marginTop: 6, lineHeight: 1.35 },
  score: { color: "#89B7CF", fontSize: 12, marginTop: 4 },

  sectionTitleRow: { display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 },
  sectionTitle: { fontWeight: 900, fontSize: 20 },
  chipRow: { display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 },
  chip: (active: boolean): React.CSSProperties => ({
    padding: "8px 12px",
    borderRadius: 999,
    border: active ? `1px solid ${TEAL}` : "1px solid rgba(255,255,255,0.12)",
    color: active ? TEAL : "#CFE9F6",
    cursor: "pointer",
    background: active ? "rgba(25,255,210,0.12)" : "transparent",
    fontWeight: 700,
    fontSize: 13.5,
    boxShadow: active ? `0 0 16px ${TEAL}22` : "none",
  }),
}

const api = {
  chat: async (message: string) => {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    })
    if (!r.ok) throw new Error("chat failed")
    return r.json()
  },
  textSearch: async (query: string, k = 12) => {
    const r = await fetch("/api/search_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, k }),
    })
    if (!r.ok) throw new Error("search failed")
    return r.json()
  },
  catalog: async () => {
    const r = await fetch("/api/catalog")
    if (!r.ok) throw new Error("catalog failed")
    return r.json()
  },
  imageSearchUpload: async (file: File, k = 12) => {
    const fd = new FormData()
    fd.append("file", file)
    fd.append("k", String(k))
    const r = await fetch("/api/search_image", { method: "POST", body: fd })
    if (!r.ok) throw new Error("image upload failed")
    return r.json()
  },
  imageSearchByUrl: async (url: string, k = 12) => {
    const r = await fetch("/api/search_by_url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, k }),
    })
    if (!r.ok) throw new Error("image url failed")
    return r.json()
  },
}

const cdn = (p: string) => `/data/${p}`

export default function App() {
  const [query, setQuery] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState("")
  const [reply, setReply] = useState("")
  const [recs, setRecs] = useState<Item[]>([])
  const [catalog, setCatalog] = useState<Item[]>([])
  const [loading, setLoading] = useState(false)
  const [catFilter, setCatFilter] = useState<string | "all">("all")

  // initial load
  useEffect(() => {
    (async () => {
      const items: Item[] = await api.catalog()
      setCatalog(items || [])
      setRecs((items || []).slice(0, 12))
      setReply("Discover popular picks below. Ask for anything or add an image for visual search.")
    })()
  }, [])

  const categories = useMemo(() => {
    const m = new Map<string, number>()
    for (const it of catalog) m.set(it.category, (m.get(it.category) || 0) + 1)
    return Array.from(m.entries()).sort((a, b) => a[0].localeCompare(b[0]))
  }, [catalog])

  const filteredCatalog = useMemo(() => {
    if (catFilter === "all") return catalog
    return catalog.filter((it) => it.category === catFilter)
  }, [catalog, catFilter])

  const onAsk = async () => {
    if (!query.trim() && !file && !imageUrl.trim()) return
    setLoading(true)
    setReply("Thinking…")
    setRecs([])
    try {
      // Image search paths
      if (file) {
        const res = await api.imageSearchUpload(file, 12)
        setReply("Here are visually similar items:")
        setRecs(res.results || [])
        return
      }
      if (imageUrl.trim()) {
        const res = await api.imageSearchByUrl(imageUrl.trim(), 12)
        setReply("Here are visually similar items:")
        setRecs(res.results || [])
        return
      }

      // Chat first — use agent results if it recommends; otherwise fallback to plain search
      const chatRes = await api.chat(query.trim())
      setReply(chatRes.text || chatRes.reply || "")

      const agentRecommended =
        chatRes &&
        typeof chatRes === "object" &&
        chatRes.intent === "recommend" &&
        Array.isArray(chatRes.results)

      if (agentRecommended) {
        setRecs(chatRes.results || [])
      } else {
        const recRes = await api.textSearch(query.trim(), 12)
        setRecs(recRes.results || [])
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.wrap}>
        {/* Header */}
        <div style={styles.header}>
          <div style={styles.logoBox}>
            <img src={logo} alt="Mercury logo" style={styles.logo} />
          </div>
          <div style={styles.titleBlock}>
            <div style={styles.title}>Mercury <span style={styles.ai}>AI</span> Commerce Agent</div>
            <div style={styles.tagline}>
              Chat, text recommendations, and image-based search — unified. Mercury routes your request to the right tool,
              combines results, and explains them in plain language.
            </div>
          </div>
          <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" style={styles.docsBtn}>
            API Docs ↗
          </a>
        </div>

        {/* Agentic input */}
        <div style={styles.panel}>
          <div style={{ fontWeight: 900, fontSize: 18, marginBottom: 8 }}>Ask anything</div>
          <div style={styles.inputRow}>
            <input
              style={styles.textInput}
              placeholder="e.g., show me a black jacket under $120 • or paste an image URL below"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <button style={styles.primaryBtn} onClick={onAsk} disabled={loading}>
              {loading ? "Working..." : "Ask Mercury"}
            </button>
            <button
              style={styles.ghostBtn}
              onClick={() => { setQuery(""); setFile(null); setImageUrl(""); }}
            >
              Clear
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, marginTop: 14 }}>
            <div style={{ ...styles.panel, background: "rgba(18, 26, 56, 0.8)" }}>
              <div style={{ color: MUTED, fontSize: 14, marginBottom: 8 }}>Upload an image</div>
              <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file && <div style={{ color: MUTED, fontSize: 12, marginTop: 8 }}>Selected: {file.name}</div>}
            </div>
            <div style={{ ...styles.panel, background: "rgba(18, 26, 56, 0.8)" }}>
              <div style={{ color: MUTED, fontSize: 14, marginBottom: 8 }}>…or paste image URL</div>
              <input
                style={styles.textInput}
                placeholder="https://example.com/photo.jpg"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
              />
            </div>
          </div>
        </div>

        {/* Agent reply */}
        {reply && (
          <div style={styles.panel}>
            <div style={styles.sectionTitleRow}>
              <div style={styles.sectionTitle}>Agent</div>
            </div>
            <div style={styles.textarea}>{reply}</div>
          </div>
        )}

        {/* Recommendations */}
        {recs.length > 0 && (
          <div style={styles.panel}>
            <div style={styles.sectionTitleRow}>
              <div style={styles.sectionTitle}>Recommendations</div>
            </div>
            <div style={styles.grid}>
              {recs.map((it) => (
                <div key={it.id} style={styles.card}>
                  <div style={styles.shot}>
                    <img src={cdn(it.image_path)} alt={it.title} style={{ maxWidth: "100%", maxHeight: "100%" }} />
                  </div>
                  <div style={{ marginTop: 12, fontWeight: 900, fontSize: 17 }}>{it.title}</div>
                  <div style={styles.meta}>{it.category} • {it.color}</div>
                  <div style={styles.price}>${it.price.toFixed(2)}</div>
                  <div style={styles.desc}>{it.description}</div>
                  {typeof it.score === "number" && <div style={styles.score}>score: {it.score.toFixed(3)}</div>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Catalog */}
        <div style={styles.panel}>
          <div style={styles.sectionTitleRow}>
            <div style={styles.sectionTitle}>Catalog</div>
            <div style={styles.chipRow}>
              <div style={styles.chip(catFilter === "all")} onClick={() => setCatFilter("all")}>All ({catalog.length})</div>
              {categories.map(([cat, count]) => (
                <div key={cat} style={styles.chip(catFilter === cat)} onClick={() => setCatFilter(cat)}>
                  {cat} ({count})
                </div>
              ))}
            </div>
          </div>
          <div style={styles.grid}>
            {filteredCatalog.map((it) => (
              <div key={it.id} style={styles.card}>
                <div style={styles.shot}>
                  <img src={cdn(it.image_path)} alt={it.title} style={{ maxWidth: "100%", maxHeight: "100%" }} />
                </div>
                <div style={{ marginTop: 12, fontWeight: 900, fontSize: 17 }}>{it.title}</div>
                <div style={styles.meta}>{it.category} • {it.color}</div>
                <div style={styles.price}>${it.price.toFixed(2)}</div>
                <div style={styles.desc}>{it.description}</div>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
