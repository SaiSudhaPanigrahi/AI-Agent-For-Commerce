import { useMemo, useState } from 'react'

const CATEGORY_FALLBACK: Record<string, string> = {
  't-shirts': 'https://images.unsplash.com/photo-1516826957135-700dedea698c?auto=format&fit=crop&w=1200&q=80',
  shoes:     'https://images.unsplash.com/photo-1608231387042-66d1773070a5?auto=format&fit=crop&w=1200&q=80',
  shorts:    'https://images.unsplash.com/photo-1603252109334-43d2011b4dde?auto=format&fit=crop&w=1200&q=80',
  hoodies:   'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=1200&q=80',
  tanks:     'https://images.unsplash.com/photo-1598978224661-5cf9ddb4ba44?auto=format&fit=crop&w=1200&q=80',
  leggings:  'https://images.unsplash.com/photo-1549570652-97324981a6fd?auto=format&fit=crop&w=1200&q=80',
  bags:      'https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=1200&q=80',
  bottles:   'https://images.unsplash.com/photo-1598986646512-3f2b36ebd9d8?auto=format&fit=crop&w=1200&q=80',
};
const GENERIC_1 = 'https://images.unsplash.com/photo-1542089363-efb2235ef2df?auto=format&fit=crop&w=1200&q=80';
const GENERIC_2 = 'https://picsum.photos/seed/mercury-gear/1200/800';

export default function ProductCard({ p, compact = false }: { p: any, compact?: boolean }) {
  const catKey = String(p.category || '').toLowerCase();
  const chain = useMemo(() => {
    const arr: string[] = [];
    if (p.image_url) arr.push(p.image_url);
    if (CATEGORY_FALLBACK[catKey]) arr.push(CATEGORY_FALLBACK[catKey]);
    arr.push(GENERIC_1, GENERIC_2);
    return arr;
  }, [p.image_url, catKey]);

  const [idx, setIdx] = useState(0);
  const [show, setShow] = useState(true);

  const cls = compact ? 'h-28' : 'h-44';
  const src = chain[idx];

  function handleError() {
    if (idx < chain.length - 1) setIdx(i => i + 1);
    else setShow(false);
  }

  return (
    <div className={`card p-3 hover:shadow-cyan-500/20 hover:border-cyan-500/30 transition ${compact ? 'text-sm' : ''}`}>
      {show && src ? (
        <div className={`relative w-full ${cls} overflow-hidden rounded-xl ring-1 ring-slate-800`}>
          <img
            src={src}
            alt={p.title}
            className={`w-full ${cls} object-cover rounded-xl`}
            loading="lazy"
            onError={handleError}
          />
        </div>
      ) : null}

      <div className="mt-2 space-y-1">
        <div className="text-slate-400">{p.brand} · {p.category}</div>
        <div className="font-semibold">{p.title}</div>
        <div className="text-slate-300">{p.description}</div>
        <div className="font-bold text-cyan-400">${Number(p.price).toFixed(2)}</div>
      </div>
    </div>
  );
}
