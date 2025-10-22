export type Product = {
  id: string;
  title: string;
  brand: string;
  category: "bags" | "shoes" | "jackets" | "caps" | "tops" | "pants" | "dresses";
  color?: string | null;
  price: number;
  description: string;
  image: string; // relative, e.g. /images/bag1.jpg
};

export type CatalogResponse = {
  items: Product[];
};
