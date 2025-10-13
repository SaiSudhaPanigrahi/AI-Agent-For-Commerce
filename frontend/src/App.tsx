import { useEffect, useState } from 'react'
import { chat, recommend, imageSearch, catalog } from './lib/api'
import ProductCard from './components/ProductCard'

export default function App() {
  const [cat, setCat] = useState<any[]>([])
  const [recs, setRecs] = useState<any[]>([])

  useEffect(() => {
    ;(async () => {
      const c = await catalog()
      setCat(c.items || [])
      const r = await recommend('lightweight sports tee under $30')
      setRecs(r.items || [])
    })()
  }, [])

  return (
    <>
      <SiteHeader />
      <main className="px-5 py-8 max-w-7xl mx-auto space-y-10">
        <section className="space-y-6">
          <ChatBox />
          <ImageSearchBox />
        </section>

        <Section title="Recommended for you">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {recs.map((p, i) => <ProductCard key={i} p={p} />)}
          </div>
        </Section>

        <Section title="Catalog">
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {cat.map((p, i) => <ProductCard key={i} p={p} />)}
          </div>
        </Section>
      </main>
    </>
  )
}

function SiteHeader() {
  return (
    <div className="site-header">
      <div className="site-header-inner">
        <div className="max-w-3xl">
          <h1 className="text-3xl md:text-4xl font-bold tracking-tight">
            Mercury <span className="text-cyan-400">AI</span> Commerce Agent
          </h1>
          <p className="text-slate-400 mt-1">
            Single agent for a commerce site: <span className="text-slate-200">chat</span>, <span className="text-slate-200">text recommendations</span>, and <span className="text-slate-200">image-based search</span> over a curated catalog.
          </p>
          <p className="text-slate-400">
            CPU-friendly local retrieval (MiniLM + CLIP) with optional OpenAI chat. FastAPI APIs and a React (Vite + Tailwind) UI.
          </p>
        </div>
        <a
          className="btn-outline whitespace-nowrap mt-1"
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noreferrer"
          title="Open FastAPI Swagger UI"
        >
          API Docs ↗
        </a>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string, children: any }) {
  return (
    <section className="space-y-3">
      <div className="text-lg font-semibold">{title}</div>
      {children}
    </section>
  )
}

type ChatLine = { you?: boolean, text?: string, items?: any[] }

function ChatBox() {
  const [text, setText] = useState('')
  const [lines, setLines] = useState<ChatLine[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function send() {
    if (!text) return
    setLoading(true); setError('')
    try {
      setLines(l => [...l, { you: true, text }])
      const res = await chat(text)
      setLines(l => [...l, { text: res.reply, items: res.items || [] }])
      setText('')
    } catch {
      setError('Failed to reach chat API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card p-5">
      <div className="text-xl font-semibold mb-3">Chat</div>
      <div className="chat-log space-y-3" style={{ minHeight: 160, maxHeight: 520 }}>
        {lines.length === 0 ? (
          <div className="text-slate-400">
            Try: <span className="text-slate-200">“tshirts under $30”</span> · <span className="text-slate-200">“trail sneakers for mud”</span> · paste an image URL
          </div>
        ) : (
          lines.map((ln, i) => (
            <div key={i} className="space-y-2">
              {ln.text && (
                <div className={ln.you ? 'font-medium text-cyan-300' : 'text-slate-200'} style={{whiteSpace:'pre-wrap'}}>
                  {ln.you ? `You: ${ln.text}` : ln.text}
                </div>
              )}
              {ln.items && ln.items.length > 0 && (
                <div className="mt-1 grid grid-cols-2 md:grid-cols-3 gap-3">
                  {ln.items.map((p:any, j:number) => <ProductCard key={j} p={p} compact />)}
                </div>
              )}
            </div>
          ))
        )}
      </div>
      <div className="flex gap-2 mt-4">
        <input className="input flex-1" value={text} onChange={e => setText(e.target.value)} placeholder="Ask for items or paste an image URL..." />
        <button className="btn" onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
      {error && <div className="text-xs text-red-400 mt-2">{error}</div>}
    </div>
  )
}

function ImageSearchBox() {
  const [url, setUrl] = useState('')
  const [items, setItems] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function go() {
    if (!url) return
    setLoading(true); setError('')
    try {
      const res = await imageSearch(url)
      setItems(res.items || [])
    } catch {
      setError('Failed to reach image search API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card p-5">
      <div className="text-xl font-semibold mb-3">Image-Based Search</div>
      <div className="flex gap-2 mb-3">
        <input className="input flex-1" value={url} onChange={e => setUrl(e.target.value)} placeholder="Paste an image URL..." />
        <button className="btn" onClick={go} disabled={loading}>{loading ? 'Searching…' : 'Search'}</button>
      </div>
      {error && <div className="text-xs text-red-400 mb-2">{error}</div>}
      {items.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {items.map((p, i) => <ProductCard key={i} p={p} />)}
        </div>
      )}
    </div>
  )
}
