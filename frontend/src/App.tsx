import { useEffect, useState } from 'react'
import { chat, recommend, imageSearch, catalog } from './lib/api'
import ProductCard from './components/ProductCard'

export default function App() {
  const [cat, setCat] = useState<any[]>([])
  const [recs, setRecs] = useState<any[]>([])

  useEffect(() => {
    (async () => {
      const c = await catalog()
      setCat(c.items || [])
      const r = await recommend('lightweight sports tee under $30')
      setRecs(r.items || [])
    })()
  }, [])

  return (
    <main className="container px-4 py-10 space-y-8">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Mercury — AI Commerce Agent</h1>
          <p className="text-slate-600">One agent: chat · text recommend · image search</p>
        </div>
        <a className="text-sm underline text-slate-600" href="http://localhost:8000/docs" target="_blank">API Docs</a>
      </header>

      <Hero />

      <section className="grid md:grid-cols-2 gap-6">
        <ChatBox />
        <ImageSearchBox />
      </section>

      <Section title="Recommended for you">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {recs.map((p, i) => <ProductCard key={i} p={p} />)}
        </div>
      </Section>

      <Section title="Catalog">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {cat.map((p, i) => <ProductCard key={i} p={p} />)}
        </div>
      </Section>

      <footer className="text-xs text-slate-500 pt-6">Demo only. No checkout. Images from Unsplash.</footer>
    </main>
  )
}

function Hero() {
  return (
    <div className="card bg-gradient-to-r from-slate-900 to-slate-700 text-white p-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <div className="text-sm uppercase tracking-widest opacity-80">Take‑Home Exercise</div>
          <div className="text-xl md:text-2xl font-semibold mt-1">Unified AI Agent for Commerce</div>
          <p className="opacity-90 mt-2 max-w-2xl">
            Chat naturally, get text-based recommendations, or paste an image URL to find visually similar products — all limited to a curated catalog.
          </p>
        </div>
        <div className="flex gap-2">
          <Badge>FastAPI</Badge>
          <Badge>React + Vite</Badge>
          <Badge>Tailwind</Badge>
          <Badge>MiniLM</Badge>
          <Badge>CLIP</Badge>
        </div>
      </div>
    </div>
  )
}

function Badge({ children }: { children: any }) {
  return <span className="bg-white/10 text-white border border-white/20 rounded-full px-3 py-1 text-xs">{children}</span>
}

function Section({ title, children }: { title: string, children: any }) {
  return (
    <section className="space-y-3">
      <div className="text-lg font-semibold">{title}</div>
      {children}
    </section>
  )
}

function ChatBox() {
  const [text, setText] = useState('')
  const [lines, setLines] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function send() {
    if (!text) return
    setLoading(true); setError('')
    try {
      const res = await chat(text)
      setLines(l => [...l, 'You: ' + text, `Agent (${res.mode}): ` + res.reply])
      setText('')
    } catch (e: any) {
      setError('Failed to reach chat API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card bg-white p-4">
      <div className="text-lg font-semibold mb-2">Chat</div>
      <div className="h-48 overflow-y-auto bg-slate-50 rounded p-2 text-sm mb-2">
        {lines.map((t, i) => <div key={i} className="mb-1">{t}</div>)}
      </div>
      <div className="flex gap-2">
        <input className="flex-1 border rounded-xl px-3 py-2" value={text} onChange={e => setText(e.target.value)} placeholder="Say hi or ask what I can do..." />
        <button className="btn bg-black text-white" onClick={send} disabled={loading}>{loading ? '...' : 'Send'}</button>
      </div>
      {error && <div className="text-xs text-red-600 mt-2">{error}</div>}
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
    } catch (e: any) {
      setError('Failed to reach image search API')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card bg-white p-4">
      <div className="text-lg font-semibold mb-2">Image-Based Search</div>
      <div className="flex gap-2 mb-3">
        <input className="flex-1 border rounded-xl px-3 py-2" value={url} onChange={e => setUrl(e.target.value)} placeholder="Paste an image URL..." />
        <button className="btn bg-black text-white" onClick={go} disabled={loading}>{loading ? 'Searching...' : 'Search'}</button>
      </div>
      {error && <div className="text-xs text-red-600 mb-2">{error}</div>}
      {items.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {items.map((p, i) => <ProductCard key={i} p={p} />)}
        </div>
      )}
    </div>
  )
}
