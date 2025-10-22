import React from "react";

type Product = {
  id: string;
  title: string;
  brand: string;
  category: "bags" | "shoes" | "jackets" | "caps" | "tops" | "pants" | "dresses";
  color?: string;
  price: number;
  description: string;
  image: string; // relative URL like /images/bag1.jpg
};

export default function ProductCard({ p }: { p: Product }) {
  return (
    <div className="bg-slate-900/40 rounded-2xl p-4 border border-slate-800 hover:border-slate-700 transition">
      <div className="aspect-square w-full overflow-hidden rounded-xl mb-3 border border-slate-800">
        <img
          src={`${import.meta.env.VITE_IMAGE_BASE}${p.image}`}
          alt={p.title}
          className="w-full h-full object-cover"
          loading="lazy"
        />
      </div>
      <div className="text-xs text-slate-400">
        <span className="uppercase tracking-wide">{p.brand}</span>
        <span className="mx-2">•</span>
        <span className="capitalize">{p.category}</span>
        {p.color ? (
          <>
            <span className="mx-2">•</span>
            <span className="capitalize">{p.color}</span>
          </>
        ) : null}
      </div>
      <div className="text-slate-100 font-medium mt-1">{p.title}</div>
      <div className="text-slate-400 text-sm mt-1 line-clamp-2">
        {p.description}
      </div>
      <div className="text-teal-300 font-semibold mt-2">
        ${p.price.toFixed(2)}
      </div>
    </div>
  );
}
