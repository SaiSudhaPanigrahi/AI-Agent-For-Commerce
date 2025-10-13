export default function ProductCard({ p }: { p: any }) {
  return (
    <div className="card bg-white p-3 hover:shadow-lg transition">
      <div className="relative w-full h-40 overflow-hidden rounded-xl">
        <img src={p.image_url} alt={p.title} className="w-full h-40 object-cover rounded-xl" />
      </div>
      <div className="mt-2 space-y-1">
        <div className="text-sm text-slate-500">{p.brand} · {p.category}</div>
        <div className="font-semibold">{p.title}</div>
        <div className="text-sm text-slate-700">{p.description}</div>
        <div className="font-bold">${p.price.toFixed(2)}</div>
      </div>
    </div>
  )
}
