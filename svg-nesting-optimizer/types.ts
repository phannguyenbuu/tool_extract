export interface SVGShapeData {
  id: string;
  name: string;
  viewBox: string; // The original full viewbox
  bbox: { x: number; y: number; w: number; h: number }; // The tight bounding box of the cut path
  widthMM: number; // Physical width of the CUT PATH
  heightMM: number; // Physical height of the CUT PATH
  path: string; // The collision path (black silhouette)
  svgContent: string; // The original inner HTML of the SVG (for rendering images/colors)
  color: string;
  strokeWidth: number;
  ratio: number; // Percentage value (0-100)
  quantity: number; // Calculated or user defined count
  scale: number; // User defined scaling percentage (default 100)
}

export interface PackedItem {
  shapeId: string;
  x: number;
  y: number;
  width: number;
  height: number;
  rotation: number;
}

export interface PackingResult {
  pages: PackedItem[][]; // Array of pages, each page has items
  efficiency: number; // Average efficiency
  counts: Record<string, number>;
  totalArea: number;
  usedArea: number;
  pageCount: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}