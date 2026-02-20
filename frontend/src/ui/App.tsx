import React, { useEffect, useMemo, useState } from "react"
import logo from "./logo.svg"

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

type ExplainContext = {
  query: string
  categoryHint: string | null
  colorHint: string | null
  minPrice: number | null
  maxPrice: number | null
}

const PROMPT_CHIPS = [
  "Show me pink shoes under $90",
  "Find black jackets for winter",
  "Travel bags under $80",
  "What can you do?",
  "Show caps in blue color",
]

const HISTORY_KEY = "mercury_query_history_v1"
const COMPARE_ITEMS_KEY = "mercury_compare_items_v1"

const BLUE = "#19E3FF"
const TEAL = "#19FFD2"
const BG0 = "#0E1430"
const PANEL_BG = "rgba(20, 28, 60, 0.88)"
const CARD_BG = "rgba(16, 24, 50, 0.94)"
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
  titleBlock: { display: "flex", flexDirection: "column" },
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
    whiteSpace: "nowrap",
  },
  actionRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    flexWrap: "wrap",
    justifyContent: "flex-end",
  },
  actionBtn: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    background: "transparent",
    border: `1px solid ${BLUE}`,
    color: BLUE,
    height: 44,
    padding: "0 14px",
    borderRadius: 12,
    fontWeight: 700,
    cursor: "pointer",
    boxShadow: `0 0 16px ${BLUE}22`,
    textDecoration: "none",
    whiteSpace: "nowrap",
  },
  actionBtnPrimary: {
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    background: BLUE,
    border: `1px solid ${BLUE}`,
    color: "#001018",
    height: 44,
    padding: "0 14px",
    borderRadius: 12,
    fontWeight: 800,
    cursor: "pointer",
    boxShadow: `0 0 20px ${BLUE}33`,
    textDecoration: "none",
    whiteSpace: "nowrap",
  },
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
  smallGhostBtn: {
    background: "transparent",
    color: BLUE,
    border: `1px solid ${BLUE}`,
    height: 34,
    padding: "0 10px",
    borderRadius: 10,
    fontWeight: 700,
    cursor: "pointer",
    fontSize: 12.5,
  },
  textarea: {
    width: "100%",
    minHeight: 120,
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
  meta: { color: "#D1E6F6", fontSize: 14, marginTop: 2, opacity: 0.9 },
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
  reasonBox: {
    marginTop: 10,
    borderRadius: 10,
    padding: 10,
    background: "rgba(8,14,33,.75)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  reasonLine: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: 12.5,
    color: "#B7CFE2",
    marginTop: 4,
  },
  compareTray: {
    marginTop: 24,
    borderRadius: 16,
    padding: 16,
    background: "rgba(9,16,36,0.92)",
    border: "1px solid rgba(25,227,255,0.25)",
    boxShadow: "0 12px 32px rgba(0,0,0,.35)",
  },
  compareGrid: {
    marginTop: 12,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
    gap: 14,
  },
  compareCard: {
    borderRadius: 12,
    padding: 12,
    background: "rgba(16,24,50,.95)",
    border: "1px solid rgba(255,255,255,0.08)",
  },
  compareRow: {
    display: "flex",
    justifyContent: "space-between",
    marginTop: 6,
    fontSize: 13,
    color: "#D7E8F4",
  },
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
  textSearch: async (
    query: string,
    k = 12,
    filters?: { category?: string; minPrice?: number; maxPrice?: number },
  ) => {
    const body: Record<string, unknown> = { query, k }
    if (filters?.category) body.category = filters.category
    if (typeof filters?.minPrice === "number") body.min_price = filters.minPrice
    if (typeof filters?.maxPrice === "number") body.max_price = filters.maxPrice
    const r = await fetch("/api/search_text", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
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

const clamp = (n: number, min: number, max: number) => Math.max(min, Math.min(max, n))

const normalizeColor = (c: string | null) => {
  if (!c) return null
  const low = c.toLowerCase().trim()
  if (low === "grey") return "gray"
  return low
}

const inferCategory = (query: string): string | null => {
  const q = query.toLowerCase()
  if (q.includes("shoe")) return "shoes"
  if (q.includes("jacket") || q.includes("coat")) return "jackets"
  if (q.includes("bag") || q.includes("tote")) return "bags"
  if (q.includes("cap") || q.includes("hat")) return "caps"
  return null
}

const inferColor = (query: string): string | null => {
  const colors = ["red", "blue", "green", "black", "white", "yellow", "brown", "gray", "grey", "purple", "pink", "orange"]
  const q = query.toLowerCase()
  for (const c of colors) {
    if (new RegExp(`\\b${c}\\b`).test(q)) return normalizeColor(c)
  }
  return null
}

const parseBudget = (query: string): { minPrice: number | null; maxPrice: number | null } => {
  const q = query.toLowerCase()
  const between = q.match(/between\s*\$?(\d+(?:\.\d+)?)\s*(?:and|to)\s*\$?(\d+(?:\.\d+)?)/)
  if (between) {
    const a = Number(between[1])
    const b = Number(between[2])
    return { minPrice: Math.min(a, b), maxPrice: Math.max(a, b) }
  }
  const under = q.match(/(?:under|below|less than)\s*\$?(\d+(?:\.\d+)?)/)
  if (under) return { minPrice: null, maxPrice: Number(under[1]) }
  const over = q.match(/(?:over|above|more than)\s*\$?(\d+(?:\.\d+)?)/)
  if (over) return { minPrice: Number(over[1]), maxPrice: null }
  return { minPrice: null, maxPrice: null }
}

const buildWhyResult = (item: Item, ctx: ExplainContext) => {
  const semantic = clamp(typeof item.score === "number" ? (item.score + 1) / 2 : 0.66, 0, 1)
  const categoryMatch = ctx.categoryHint ? item.category.toLowerCase() === ctx.categoryHint.toLowerCase() : false
  const colorMatch = ctx.colorHint ? normalizeColor(item.color) === normalizeColor(ctx.colorHint) : false
  const inBudget =
    (ctx.minPrice === null || item.price >= ctx.minPrice) &&
    (ctx.maxPrice === null || item.price <= ctx.maxPrice)
  const colorBonus = colorMatch ? 0.12 : 0
  const categoryBonus = categoryMatch ? 0.12 : 0
  const budgetBonus = inBudget ? 0.08 : -0.04
  const total = clamp(semantic + colorBonus + categoryBonus + budgetBonus, 0, 1.5)
  return {
    categoryMatch,
    colorMatch,
    inBudget,
    semantic: semantic.toFixed(2),
    colorBonus: colorBonus.toFixed(2),
    categoryBonus: categoryBonus.toFixed(2),
    budgetBonus: budgetBonus.toFixed(2),
    total: total.toFixed(2),
  }
}

export default function App() {
  const [query, setQuery] = useState("")
  const [file, setFile] = useState<File | null>(null)
  const [imageUrl, setImageUrl] = useState("")
  const [reply, setReply] = useState("")
  const [recs, setRecs] = useState<Item[]>([])
  const [catalog, setCatalog] = useState<Item[]>([])
  const [loading, setLoading] = useState(false)
  const [compareIds, setCompareIds] = useState<string[]>([])
  const [history, setHistory] = useState<string[]>([])
  const [explainContext, setExplainContext] = useState<ExplainContext>({
    query: "",
    categoryHint: null,
    colorHint: null,
    minPrice: null,
    maxPrice: null,
  })

  useEffect(() => {
    try {
      const raw = localStorage.getItem(HISTORY_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed)) setHistory(parsed.filter((x) => typeof x === "string").slice(0, 8))
    } catch {
      // ignore broken local storage payloads
    }
  }, [])

  useEffect(() => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 8)))
  }, [history])

  useEffect(() => {
    ;(async () => {
      try {
        const items: Item[] = await api.catalog()
        setCatalog(items || [])
        setRecs((items || []).slice(0, 12))
        setReply("Discover popular picks below. Ask for anything or add an image for visual search.")
      } catch {
        setReply("Unable to load catalog. Check backend and refresh.")
      }
    })()
  }, [])

  const itemById = useMemo(() => {
    const m = new Map<string, Item>()
    for (const it of catalog) m.set(it.id, it)
    for (const it of recs) m.set(it.id, it)
    return m
  }, [catalog, recs])

  const updateHistory = (q: string) => {
    const normalized = q.trim()
    if (!normalized) return
    setHistory((prev) => [normalized, ...prev.filter((x) => x !== normalized)].slice(0, 8))
  }

  const toggleCompare = (item: Item) => {
    setCompareIds((prev) => {
      if (prev.includes(item.id)) return prev.filter((id) => id !== item.id)
      if (prev.length >= 3) return prev
      return [...prev, item.id]
    })
  }

  useEffect(() => {
    const selected = compareIds.map((id) => itemById.get(id)).filter(Boolean)
    localStorage.setItem(COMPARE_ITEMS_KEY, JSON.stringify(selected))
  }, [compareIds, itemById])

  const openCompareWindow = () => {
    const selected = compareIds.map((id) => itemById.get(id)).filter(Boolean) as Item[]
    if (selected.length === 0) {
      setReply("Add at least one product to compare.")
      return
    }
    localStorage.setItem(COMPARE_ITEMS_KEY, JSON.stringify(selected))
    const popup = window.open("/compare.html", "mercury-compare-window", "width=1100,height=760,resizable=yes,scrollbars=yes")
    if (!popup) {
      setReply("Popup blocked. Allow popups for this site to open compare window.")
      return
    }
    popup.postMessage({ type: "MERCURY_COMPARE_ITEMS", payload: selected }, window.location.origin)
  }

  const openBrowseWindow = () => {
    const popup = window.open("/browse.html", "mercury-browse-window", "width=1320,height=860,resizable=yes,scrollbars=yes")
    if (!popup) {
      window.open("/browse.html", "_blank")
    }
  }

  const onAsk = async (forcedQuery?: string) => {
    const activeQuery = (forcedQuery ?? query).trim()
    if (!activeQuery && !file && !imageUrl.trim()) return

    const parsed = parseBudget(activeQuery)
    const effectiveMinPrice = parsed.minPrice
    const effectiveMaxPrice = parsed.maxPrice
    const effectiveCategory = inferCategory(activeQuery)
    const effectiveColor = inferColor(activeQuery)

    setExplainContext({
      query: activeQuery,
      categoryHint: effectiveCategory,
      colorHint: effectiveColor,
      minPrice: effectiveMinPrice,
      maxPrice: effectiveMaxPrice,
    })

    setLoading(true)
    setReply("Thinking...")
    setRecs([])
    try {
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

      const chatRes = await api.chat(activeQuery)
      setReply(chatRes.text || chatRes.reply || "")

      const agentRecommended =
        chatRes &&
        typeof chatRes === "object" &&
        chatRes.intent === "recommend" &&
        Array.isArray(chatRes.results)

      if (agentRecommended) {
        setRecs(chatRes.results || [])
      } else {
        const recRes = await api.textSearch(activeQuery, 12, {
          category: effectiveCategory || undefined,
          minPrice: effectiveMinPrice,
          maxPrice: effectiveMaxPrice,
        })
        setRecs(recRes.results || [])
      }

      updateHistory(activeQuery)
      setQuery(activeQuery)
    } catch {
      setReply("Something went wrong while searching. Try again.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.wrap}>
        <div style={styles.header}>
          <div style={styles.logoBox}>
            <img src={logo} alt="Mercury logo" style={styles.logo} />
          </div>
          <div style={styles.titleBlock}>
            <div style={styles.title}>
              Mercury <span style={styles.ai}>AI</span> Commerce Agent
            </div>
            <div style={styles.tagline}>
              Ask for recommendations with AI, then explore the full storefront and compare shortlisted items.
            </div>
          </div>
          <div style={styles.actionRow}>
            <button style={styles.actionBtnPrimary} onClick={openBrowseWindow}>
              Shop Catalog
            </button>
            <button style={styles.actionBtn} onClick={openCompareWindow}>
              Compare ({compareIds.length}/3)
            </button>
            {compareIds.length > 0 && (
              <button style={styles.actionBtn} onClick={() => setCompareIds([])}>
                Clear
              </button>
            )}
            <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer" style={styles.docsBtn}>
              API Docs ↗
            </a>
          </div>
        </div>

        <div style={styles.panel}>
          <div style={{ fontWeight: 900, fontSize: 18, marginBottom: 8 }}>Ask anything</div>
          <div style={styles.inputRow}>
            <input
              style={styles.textInput}
              placeholder="e.g., show me a black jacket under $120"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") void onAsk()
              }}
            />
            <button style={styles.primaryBtn} onClick={() => void onAsk()} disabled={loading}>
              {loading ? "Working..." : "Ask Mercury"}
            </button>
            <button
              style={styles.ghostBtn}
              onClick={() => {
                setQuery("")
                setFile(null)
                setImageUrl("")
              }}
            >
              Clear
            </button>
          </div>

          <div style={styles.chipRow}>
            {PROMPT_CHIPS.map((chip) => (
              <button
                key={chip}
                style={styles.chip(false)}
                onClick={() => {
                  setQuery(chip)
                  void onAsk(chip)
                }}
              >
                {chip}
              </button>
            ))}
          </div>

          {history.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div style={{ color: MUTED, fontSize: 13.5 }}>Session history (click to rerun)</div>
              <div style={styles.chipRow}>
                {history.map((h) => (
                  <button
                    key={h}
                    style={styles.chip(false)}
                    onClick={() => {
                      setQuery(h)
                      void onAsk(h)
                    }}
                    title={h}
                  >
                    {h.length > 32 ? `${h.slice(0, 32)}...` : h}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 18, marginTop: 14 }}>
            <div style={{ ...styles.panel, background: "rgba(18, 26, 56, 0.8)", marginTop: 0 }}>
              <div style={{ color: MUTED, fontSize: 14, marginBottom: 8 }}>Upload an image</div>
              <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
              {file && <div style={{ color: MUTED, fontSize: 12, marginTop: 8 }}>Selected: {file.name}</div>}
            </div>
            <div style={{ ...styles.panel, background: "rgba(18, 26, 56, 0.8)", marginTop: 0 }}>
              <div style={{ color: MUTED, fontSize: 14, marginBottom: 8 }}>...or paste image URL</div>
              <input
                style={styles.textInput}
                placeholder="https://example.com/photo.jpg"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
              />
            </div>
          </div>
        </div>

        {reply && (
          <div style={styles.panel}>
            <div style={styles.sectionTitleRow}>
              <div style={styles.sectionTitle}>Agent</div>
            </div>
            <div style={styles.textarea}>{reply}</div>
          </div>
        )}

        {recs.length > 0 && (
          <div style={styles.panel}>
            <div style={styles.sectionTitleRow}>
              <div style={styles.sectionTitle}>Recommendations ({recs.length})</div>
            </div>
            <div style={styles.grid}>
              {recs.map((it) => {
                const why = buildWhyResult(it, explainContext)
                const isCompared = compareIds.includes(it.id)
                return (
                  <div key={it.id} style={styles.card}>
                    <div style={styles.shot}>
                      <img src={cdn(it.image_path)} alt={it.title} style={{ maxWidth: "100%", maxHeight: "100%" }} />
                    </div>
                    <div style={{ marginTop: 12, fontWeight: 900, fontSize: 17 }}>{it.title}</div>
                    <div style={styles.meta}>
                      {it.category} • {it.color}
                    </div>
                    <div style={styles.price}>${it.price.toFixed(2)}</div>
                    <div style={styles.desc}>{it.description}</div>
                    {typeof it.score === "number" && <div style={styles.score}>raw score: {it.score.toFixed(3)}</div>}
                    <button style={styles.smallGhostBtn} onClick={() => toggleCompare(it)}>
                      {isCompared ? "Remove Compare" : "Add Compare"}
                    </button>
                    <div style={styles.reasonBox}>
                      <div style={{ fontWeight: 700, fontSize: 13.5, color: "#DDF2FF" }}>Why this result</div>
                      <div style={styles.reasonLine}>
                        <span>Category match</span>
                        <span>{why.categoryMatch ? "yes" : "no"}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Color match</span>
                        <span>{why.colorMatch ? "yes" : "no"}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Budget fit</span>
                        <span>{why.inBudget ? "yes" : "no"}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Semantic</span>
                        <span>{why.semantic}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Color bonus</span>
                        <span>{why.colorBonus}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Category bonus</span>
                        <span>{why.categoryBonus}</span>
                      </div>
                      <div style={styles.reasonLine}>
                        <span>Budget bonus</span>
                        <span>{why.budgetBonus}</span>
                      </div>
                      <div style={{ ...styles.reasonLine, fontWeight: 800, marginTop: 8 }}>
                        <span>Estimated total</span>
                        <span>{why.total}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

      </div>
    </div>
  )
}
