import React, { useMemo, useRef, useState, useEffect, useCallback } from "react";
import Konva from "konva";
import { Stage, Layer, Line, Text, Circle, Rect, Path, Group, Image, Transformer } from "react-konva";

const toPoints = (pts) => pts.flatMap((p) => [p[0], p[1]]);

const toFinite = (v, fallback = 0) => (Number.isFinite(v) ? v : fallback);
const escapeXml = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");

const measureText = (text, fontSize, fontFamily) => {
  const size = toFinite(fontSize, 12);
  const family = fontFamily || "Arial";
  const safeText = String(text ?? "");
  if (Konva?.Util?.getTextWidth) {
    const widthRaw = Konva.Util.getTextWidth(safeText, size, family);
    const widthFallback = safeText.length * size * 0.6;
    return { width: toFinite(widthRaw, widthFallback), height: toFinite(size, 12) };
  }
  const width = safeText.length * size * 0.6;
  return { width: toFinite(width, 0), height: toFinite(size, 12) };
};
const rotatePt = (pt, angleDeg, cx, cy) => {
  if (!angleDeg) return pt;
  const ang = (angleDeg * Math.PI) / 180;
  const c = Math.cos(ang);
  const s = Math.sin(ang);
  const x = pt[0] - cx;
  const y = pt[1] - cy;
  return [cx + x * c - y * s, cy + x * s + y * c];
};

const transformPath = (pts, shift, rot, center) => {
  if (!pts || !pts.length) return [];
  const dx = shift?.[0] ?? 0;
  const dy = shift?.[1] ?? 0;
  const ang = rot ?? 0;
  const cx = center?.[0] ?? 0;
  const cy = center?.[1] ?? 0;
  return pts.map((p) => {
    const r = rotatePt(p, ang, cx, cy);
    return [r[0] + dx, r[1] + dy];
  });
};

const bboxFromPts = (pts) => {
  if (!pts || !pts.length) return null;
  let minx = pts[0][0];
  let maxx = pts[0][0];
  let miny = pts[0][1];
  let maxy = pts[0][1];
  for (let i = 1; i < pts.length; i++) {
    const x = pts[i][0];
    const y = pts[i][1];
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }
  return { minx, maxx, miny, maxy };
};

const polyCentroid = (poly) => {
  if (!poly || poly.length < 3) return { area: 0, cx: 0, cy: 0 };
  let area = 0;
  let cx = 0;
  let cy = 0;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % n];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  area *= 0.5;
  if (Math.abs(area) < 1e-6) return { area: 0, cx: poly[0][0], cy: poly[0][1] };
  return { area, cx: cx / (6 * area), cy: cy / (6 * area) };
};

const pointInPoly = (pt, poly) => {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    const intersect = yi > pt[1] !== yj > pt[1] &&
      pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi + 0.0) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
};

const pointSegDist = (pt, a, b) => {
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const wx = pt[0] - a[0];
  const wy = pt[1] - a[1];
  const c1 = vx * wx + vy * wy;
  if (c1 <= 0) return Math.hypot(pt[0] - a[0], pt[1] - a[1]);
  const c2 = vx * vx + vy * vy;
  if (c2 <= c1) return Math.hypot(pt[0] - b[0], pt[1] - b[1]);
  const t = c1 / c2;
  const px = a[0] + t * vx;
  const py = a[1] + t * vy;
  return Math.hypot(pt[0] - px, pt[1] - py);
};

const pointInPolyWithOffset = (pt, poly, offset) => {
  if (pointInPoly(pt, poly)) return true;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    if (pointSegDist(pt, a, b) <= offset) return true;
  }
  return false;
};

const buildPackedPolyData = (data) => {
  if (!data?.regions || !data?.zone_id) return [];
  return data.regions.map((poly, rid) => {
    const zid = data.zone_id[rid];
    const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
    const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
    const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0, 0];
    const tpts = shift ? transformPath(poly, shift, rot, center) : poly;
    return { pts: tpts, bbox: bboxFromPts(tpts) };
  });
};

const buildPackedEmptyCells = (data, packedPolyData) => {
  if (!data?.canvas || !packedPolyData?.length) return [];
  const cellSize = 6;
  const radius = 3;
  const pts = [];
  const w = data.canvas.w;
  const h = data.canvas.h;
  for (let y = cellSize; y + cellSize <= h; y += cellSize) {
    for (let x = cellSize; x + cellSize <= w; x += cellSize) {
      const cx = x + radius;
      const cy = y + radius;
      if (cx < radius || cy < radius || cx > w - radius || cy > h - radius) continue;
      const corners = [
        [x, y],
        [x + cellSize, y],
        [x + cellSize, y + cellSize],
        [x, y + cellSize],
        [cx, cy],
      ];
      let inside = false;
      for (const poly of packedPolyData) {
        const bb = poly.bbox;
        if (!bb) continue;
        const minx = bb.minx - radius;
        const maxx = bb.maxx + radius;
        const miny = bb.miny - radius;
        const maxy = bb.maxy + radius;
        if (x > maxx || x + cellSize < minx || y > maxy || y + cellSize < miny) continue;
        for (const pt of corners) {
          if (pointInPolyWithOffset(pt, poly.pts, radius)) {
            inside = true;
            break;
          }
        }
        if (inside) break;
      }
      if (!inside) pts.push([cx, cy]);
    }
  }
  return pts;
};

const scalePts = (pts, ratio) => (pts || []).map((p) => [p[0] * ratio, p[1] * ratio]);

const scaleSegments = (segments, ratio) =>
  (segments || []).map((seg) => scalePts(seg, ratio));

const scaleVoronoiData = (voronoi, ratio) => {
  if (!voronoi) return { mask: [], cells: [], snappedCells: [] };
  return {
    ...voronoi,
    mask: (voronoi.mask || []).map(([x, y]) => [x * ratio, y * ratio]),
    cells: (voronoi.cells || []).map((poly) => (poly || []).map(([x, y]) => [x * ratio, y * ratio])),
    snappedCells:
      (voronoi.snappedCells || voronoi.snapped_cells || []).map((poly) =>
        (poly || []).map(([x, y]) => [x * ratio, y * ratio])
      ),
  };
};

const normalizeNodesForSave = (nodes, scale = 1) =>
  (nodes || []).map((n, idx) => ({
    id: idx,
    x: (n?.x || 0) / scale,
    y: (n?.y || 0) / scale,
  }));

const normalizeVoronoiForSave = (voronoi, scale = 1) => ({
  mask: (voronoi?.mask || []).map(([x, y]) => [x / scale, y / scale]),
  cells: (voronoi?.cells || []).map((poly) => (poly || []).map(([x, y]) => [x / scale, y / scale])),
  snapped_cells: (voronoi?.snappedCells || []).map((poly) =>
    (poly || []).map(([x, y]) => [x / scale, y / scale])
  ),
});

const cloneSourceNodes = (nodes) =>
  (nodes || []).map((n, idx) => ({
    ...(n || {}),
    id: Number.isFinite(n?.id) ? n.id : idx,
    x: Number(n?.x) || 0,
    y: Number(n?.y) || 0,
  }));

const cloneSourceSegs = (segs) => (segs || []).map(([a, b]) => [a, b]);

const cloneSourceVoronoi = (voronoi) => ({
  ...(voronoi || {}),
  mask: (voronoi?.mask || []).map(([x, y]) => [x, y]),
  cells: (voronoi?.cells || []).map((poly) => (poly || []).map(([x, y]) => [x, y])),
  snappedCells: (voronoi?.snappedCells || []).map((poly) =>
    (poly || []).map(([x, y]) => [x, y])
  ),
});

const scaleSceneData = (scene, ratio) => {
  if (!scene || ratio === 1) return scene;
  const scaleNum = (v) => (Number.isFinite(v) ? v * ratio : v);
  const scalePolyList = (polys) => (polys || []).map((poly) => scalePts(poly || [], ratio));
  const scaleLineDict = (dict) => {
    const out = {};
    Object.entries(dict || {}).forEach(([k, paths]) => {
      out[k] = (paths || []).map((path) => scalePts(path || [], ratio));
    });
    return out;
  };
  const scaleLabelDict = (dict) => {
    const out = {};
    Object.entries(dict || {}).forEach(([k, v]) => {
      if (!v) return;
      out[k] = { ...v, x: scaleNum(v.x), y: scaleNum(v.y) };
    });
    return out;
  };
  const scaleShiftDict = (dict) => {
    const out = {};
    Object.entries(dict || {}).forEach(([k, v]) => {
      if (!Array.isArray(v) || v.length < 2) return;
      out[k] = [scaleNum(v[0]), scaleNum(v[1])];
    });
    return out;
  };
  const scalePlacements = (placements) =>
    (placements || []).map((p) =>
      Array.isArray(p) && p.length >= 4
        ? [scaleNum(p[0]), scaleNum(p[1]), scaleNum(p[2]), scaleNum(p[3]), p[4]]
        : p
    );
  const scaleRotInfo = (rotInfo) =>
    (rotInfo || []).map((info) => {
      if (!info || typeof info !== "object") return info;
      const out = { ...info };
      ["cx", "cy", "minx", "miny", "maxx", "maxy"].forEach((k) => {
        if (Number.isFinite(out[k])) out[k] = out[k] * ratio;
      });
      return out;
    });

  return {
    ...scene,
    canvas: scene.canvas
      ? { ...scene.canvas, w: scaleNum(scene.canvas.w), h: scaleNum(scene.canvas.h) }
      : scene.canvas,
    regions: scalePolyList(scene.regions),
    zone_boundaries: scaleLineDict(scene.zone_boundaries),
    zone_labels: scaleLabelDict(scene.zone_labels),
    region_labels: scaleLabelDict(scene.region_labels),
    zone_pack_polys: scalePolyList(scene.zone_pack_polys),
    zone_polys: scalePolyList(scene.zone_polys),
    placements: scalePlacements(scene.placements),
    rot_info: scaleRotInfo(scene.rot_info),
    zone_shift: scaleShiftDict(scene.zone_shift),
    zone_center: scaleShiftDict(scene.zone_center),
    voronoi: scene.voronoi
      ? {
          ...scene.voronoi,
          mask: scalePolyList([scene.voronoi.mask || []])[0] || [],
          cells: scalePolyList(scene.voronoi.cells),
          snapped_cells: scalePolyList(scene.voronoi.snapped_cells),
        }
      : scene.voronoi,
  };
};

const logPackedPreview = () => {};

// packed zone boundaries are transformed on backend

const parsePoints = (str) => {
  if (!str) return [];
  const raw = str
    .trim()
    .replace(/\s+/g, " ")
    .split(" ")
    .flatMap((p) => p.split(","))
    .map((v) => v.trim())
    .filter(Boolean);
  const pts = [];
  for (let i = 0; i + 1 < raw.length; i += 2) {
    const x = parseFloat(raw[i]);
    const y = parseFloat(raw[i + 1]);
    if (Number.isFinite(x) && Number.isFinite(y)) pts.push([x, y]);
  }
  return pts;
};

const parseSvgSize = (svgText) => {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const svg = doc.querySelector("svg");
  if (!svg) return { w: 1000, h: 1000 };
  const vb = svg.getAttribute("viewBox");
  if (vb) {
    const parts = vb.replace(/,/g, " ").trim().split(/\s+/).map(parseFloat);
    if (parts.length === 4 && parts.every(Number.isFinite)) {
      return { w: parts[2], h: parts[3] };
    }
  }
  const w = parseFloat(svg.getAttribute("width") || "1000");
  const h = parseFloat(svg.getAttribute("height") || "1000");
  return {
    w: Number.isFinite(w) ? w : 1000,
    h: Number.isFinite(h) ? h : 1000,
  };
};

const getSvgHref = (el) =>
  el.getAttribute("href") || el.getAttribute("xlink:href") || el.getAttribute("href.baseVal") || "";

const svgToDataUrl = (svgText) =>
  `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;

const decodeSvgDataUrl = (src) => {
  if (!src || !src.startsWith("data:image/svg+xml")) return null;
  const parts = src.split(",");
  if (parts.length < 2) return null;
  try {
    return decodeURIComponent(parts.slice(1).join(","));
  } catch {
    return null;
  }
};

const applySvgFill = (svgText, color) => {
  if (!svgText) return svgText;
  try {
    const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    const tags = ["path", "rect", "circle", "ellipse", "polygon", "polyline"];
    tags.forEach((tag) => {
      doc.querySelectorAll(tag).forEach((el) => {
        const fill = el.getAttribute("fill");
        if (fill && fill.toLowerCase() === "none") return;
        el.setAttribute("fill", color);
      });
    });
    return new XMLSerializer().serializeToString(doc);
  } catch {
    return svgText;
  }
};

const parseOverlayItems = (svgText) => {
  if (!svgText) return [];
  try {
    const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
    const nodes = Array.from(
      doc.querySelectorAll("g#OVERLAY image[data-overlay='1'], image[data-overlay='1']")
    );
    return nodes.map((node, idx) => {
      const src = getSvgHref(node);
      const x = parseFloat(node.getAttribute("data-x") || node.getAttribute("x") || "0");
      const y = parseFloat(node.getAttribute("data-y") || node.getAttribute("y") || "0");
      const width = parseFloat(node.getAttribute("data-width") || node.getAttribute("width") || "0");
      const height = parseFloat(
        node.getAttribute("data-height") || node.getAttribute("height") || "0"
      );
      const scaleX = parseFloat(node.getAttribute("data-scale-x") || "1");
      const scaleY = parseFloat(node.getAttribute("data-scale-y") || "1");
      const rotation = parseFloat(node.getAttribute("data-rotation") || "0");
      return {
        id: node.getAttribute("data-id") || `overlay-${idx}`,
        src,
        x,
        y,
        width,
        height,
        scaleX: Number.isFinite(scaleX) ? scaleX : 1,
        scaleY: Number.isFinite(scaleY) ? scaleY : 1,
        rotation: Number.isFinite(rotation) ? rotation : 0,
      };
    });
  } catch {
    return [];
  }
};

const loadImageFromSrc = (src) =>
  new Promise((resolve) => {
    if (!src) {
      resolve(null);
      return;
    }
    const img = new window.Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });

const buildSegmentsFromSvg = (svgText) => {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const segments = [];
  const borderSegments = [];
  const svgSize = parseSvgSize(svgText);
  const isOuterBorder = (pts) => {
    if (!pts || pts.length < 4) return false;
    const xs = pts.map((p) => p[0]);
    const ys = pts.map((p) => p[1]);
    const minx = Math.min(...xs);
    const maxx = Math.max(...xs);
    const miny = Math.min(...ys);
    const maxy = Math.max(...ys);
    const tol = 1.0;
    return (
      Math.abs(minx - 0) < tol ||
      Math.abs(miny - 0) < tol ||
      Math.abs(maxx - svgSize.w) < tol ||
      Math.abs(maxy - svgSize.h) < tol
    );
  };
  doc.querySelectorAll("line").forEach((el) => {
    const x1 = parseFloat(el.getAttribute("x1") || "0");
    const y1 = parseFloat(el.getAttribute("y1") || "0");
    const x2 = parseFloat(el.getAttribute("x2") || "0");
    const y2 = parseFloat(el.getAttribute("y2") || "0");
    segments.push([[x1, y1], [x2, y2]]);
  });
  doc.querySelectorAll("polyline").forEach((el) => {
    const pts = parsePoints(el.getAttribute("points"));
    const isBorder = isOuterBorder(pts);
    for (let i = 0; i + 1 < pts.length; i++) {
      const seg = [pts[i], pts[i + 1]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
  });
  doc.querySelectorAll("polygon").forEach((el) => {
    const pts = parsePoints(el.getAttribute("points"));
    const isBorder = isOuterBorder(pts);
    for (let i = 0; i + 1 < pts.length; i++) {
      const seg = [pts[i], pts[i + 1]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
    if (pts.length > 2) {
      const seg = [pts[pts.length - 1], pts[0]];
      segments.push(seg);
      if (isBorder) borderSegments.push(seg);
    }
  });
  return { segments, borderSegments };
};

const snapNodes = (segments, snap) => {
  const cells = new Map();
  const nodes = [];
  const nodeSum = [];
  const nodeCnt = [];

  const cellKey = (x, y) => `${Math.floor(x / snap)},${Math.floor(y / snap)}`;

  const findOrCreate = (pt) => {
    const [x, y] = pt;
    if (!snap || snap <= 0) {
      const id = nodes.length;
      nodes.push({ id, x, y });
      return id;
    }
    const cx = Math.floor(x / snap);
    const cy = Math.floor(y / snap);
    let found = -1;
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        const key = `${cx + dx},${cy + dy}`;
        const ids = cells.get(key);
        if (!ids) continue;
        for (const id of ids) {
          const nx = nodes[id].x;
          const ny = nodes[id].y;
          const d2 = (nx - x) * (nx - x) + (ny - y) * (ny - y);
          if (d2 <= snap * snap) {
            found = id;
            break;
          }
        }
        if (found !== -1) break;
      }
      if (found !== -1) break;
    }
    if (found === -1) {
      const id = nodes.length;
      nodes.push({ id, x, y });
      const key = cellKey(x, y);
      if (!cells.has(key)) cells.set(key, []);
      cells.get(key).push(id);
      nodeSum[id] = [x, y];
      nodeCnt[id] = 1;
      return id;
    }
    nodeSum[found][0] += x;
    nodeSum[found][1] += y;
    nodeCnt[found] += 1;
    return found;
  };

  const segs = segments.map(([a, b]) => {
    const ai = findOrCreate(a);
    const bi = findOrCreate(b);
    return [ai, bi];
  });

  // recompute centroid positions
  nodes.forEach((n, i) => {
    if (nodeCnt[i]) {
      n.x = nodeSum[i][0] / nodeCnt[i];
      n.y = nodeSum[i][1] / nodeCnt[i];
    }
  });

  return { nodes, segs };
};

const calcBounds = (polys) => {
  if (!polys || polys.length === 0) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  polys.forEach((poly) => {
    poly.forEach((p) => {
      minx = Math.min(minx, p[0]);
      miny = Math.min(miny, p[1]);
      maxx = Math.max(maxx, p[0]);
      maxy = Math.max(maxy, p[1]);
    });
  });
  return { minx, miny, maxx, maxy };
};

const calcBoundsFromLines = (linesDict) => {
  if (!linesDict) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  let minx = Infinity;
  let miny = Infinity;
  let maxx = -Infinity;
  let maxy = -Infinity;
  let found = false;
  Object.values(linesDict).forEach((lines) => {
    (lines || []).forEach((pts) => {
      pts.forEach((p) => {
        found = true;
        minx = Math.min(minx, p[0]);
        miny = Math.min(miny, p[1]);
        maxx = Math.max(maxx, p[0]);
        maxy = Math.max(maxy, p[1]);
      });
    });
  });
  if (!found) return { minx: 0, miny: 0, maxx: 1, maxy: 1 };
  return { minx, miny, maxx, maxy };
};

const mergeNodesIfClose = (nodes, segs, movedId, snap) => {
  if (!snap || snap <= 0) return { nodes, segs };
  const moved = nodes.find((n) => n.id === movedId);
  if (!moved) return { nodes, segs };
  let targetId = null;
  for (const n of nodes) {
    if (n.id === movedId) continue;
    const dx = n.x - moved.x;
    const dy = n.y - moved.y;
    if (dx * dx + dy * dy <= snap * snap) {
      targetId = n.id;
      break;
    }
  }
  if (targetId == null) return { nodes, segs };

  const merged = nodes
    .filter((n) => n.id !== movedId)
    .map((n) =>
      n.id === targetId
        ? { ...n, x: (n.x + moved.x) / 2, y: (n.y + moved.y) / 2 }
        : n
    );

  const remap = new Map();
  merged.forEach((n, idx) => {
    remap.set(n.id, idx);
  });
  const newSegs = segs
    .map(([a, b]) => {
      const na = a === movedId ? targetId : a;
      const nb = b === movedId ? targetId : b;
      if (na === nb) return null;
      return [remap.get(na), remap.get(nb)];
    })
    .filter(Boolean);

  const newNodes = merged.map((n, idx) => ({ ...n, id: idx }));
  return { nodes: newNodes, segs: newSegs };
};

const buildVoronoiVertexGraph = (cells, eps = 0.01) => {
  const keyToId = new Map();
  const vertices = [];
  const refs = (cells || []).map((poly) =>
    (poly || []).map((pt) => {
      const key = `${(pt?.[0] ?? 0).toFixed(3)}:${(pt?.[1] ?? 0).toFixed(3)}`;
      let id = keyToId.get(key);
      if (id == null) {
        id = vertices.length;
        keyToId.set(key, id);
        vertices.push({ id, x: pt[0], y: pt[1] });
      }
      return id;
    })
  );
  return { vertices, refs, eps };
};

const rebuildVoronoiCellsFromGraph = (vertices, refs) =>
  (refs || [])
    .map((polyRefs) => {
      const pts = [];
      (polyRefs || []).forEach((vid) => {
        const v = vertices[vid];
        if (!v) return;
        const prev = pts[pts.length - 1];
        if (prev && Math.abs(prev[0] - v.x) < 1e-9 && Math.abs(prev[1] - v.y) < 1e-9) return;
        pts.push([v.x, v.y]);
      });
      if (pts.length > 1) {
        const first = pts[0];
        const last = pts[pts.length - 1];
        if (Math.abs(first[0] - last[0]) < 1e-9 && Math.abs(first[1] - last[1]) < 1e-9) {
          pts.pop();
        }
      }
      return pts.length >= 3 ? pts : null;
    })
    .filter(Boolean);

const mergeVoronoiVerticesIfClose = (vertices, refs, movedId, snap) => {
  if (!snap || snap <= 0) return { vertices, refs };
  const moved = vertices[movedId];
  if (!moved) return { vertices, refs };
  let targetId = null;
  for (const v of vertices) {
    if (!v || v.id === movedId) continue;
    const dx = v.x - moved.x;
    const dy = v.y - moved.y;
    if (dx * dx + dy * dy <= snap * snap) {
      targetId = v.id;
      break;
    }
  }
  if (targetId == null) return { vertices, refs };
  const nextVertices = vertices
    .filter((v) => v.id !== movedId)
    .map((v) =>
      v.id === targetId
        ? { ...v, x: (v.x + moved.x) / 2, y: (v.y + moved.y) / 2 }
        : { ...v }
    );
  const remap = new Map();
  nextVertices.forEach((v, idx) => {
    remap.set(v.id, idx);
  });
  const nextRefs = (refs || []).map((polyRefs) =>
    (polyRefs || []).map((vid) => {
      const nextId = vid === movedId ? targetId : vid;
      return remap.get(nextId);
    })
  );
  const normalizedVertices = nextVertices.map((v, idx) => ({ ...v, id: idx }));
  return { vertices: normalizedVertices, refs: nextRefs };
};

const projectPointToSegment = (pt, a, b) => {
  const vx = b[0] - a[0];
  const vy = b[1] - a[1];
  const ll = vx * vx + vy * vy;
  if (ll <= 1e-9) return null;
  const t = ((pt[0] - a[0]) * vx + (pt[1] - a[1]) * vy) / ll;
  if (t <= 1e-6 || t >= 1 - 1e-6) return null;
  return [a[0] + t * vx, a[1] + t * vy];
};

const dedupeGraphNodes = (nodes, segs) => {
  const keyToId = new Map();
  const remap = new Map();
  const nextNodes = [];
  nodes.forEach((n) => {
    const key = `${n.x.toFixed(3)},${n.y.toFixed(3)}`;
    let nid = keyToId.get(key);
    if (nid == null) {
      nid = nextNodes.length;
      keyToId.set(key, nid);
      nextNodes.push({ id: nid, x: n.x, y: n.y });
    }
    remap.set(n.id, nid);
  });
  const edgeSeen = new Set();
  const nextSegs = [];
  (segs || []).forEach(([a, b]) => {
    const na = remap.get(a);
    const nb = remap.get(b);
    if (na == null || nb == null || na === nb) return;
    const ek = edgeKey(na, nb);
    if (edgeSeen.has(ek)) return;
    edgeSeen.add(ek);
    nextSegs.push([na, nb]);
  });
  return { nodes: nextNodes, segs: nextSegs };
};

const projectGraphNodesToVoronoi = (nodes, segs, snappedCells, maxDist = 4) => {
  if (!nodes?.length || !segs?.length || !snappedCells?.length) return { nodes, segs };
  const zoneSegs = [];
  (snappedCells || []).forEach((poly) => {
    if (!poly || poly.length < 2) return;
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i];
      const b = poly[(i + 1) % poly.length];
      if (!a || !b) continue;
      zoneSegs.push([a, b]);
    }
  });
  if (!zoneSegs.length) return { nodes, segs };
  const nextNodes = nodes.map((n) => {
    let best = [n.x, n.y];
    let bestD2 = maxDist * maxDist;
    zoneSegs.forEach(([a, b]) => {
      const proj = projectPointToSegment([n.x, n.y], a, b);
      if (!proj) return;
      const dx = proj[0] - n.x;
      const dy = proj[1] - n.y;
      const d2 = dx * dx + dy * dy;
      if (d2 <= bestD2) {
        bestD2 = d2;
        best = proj;
      }
    });
    return { ...n, x: best[0], y: best[1] };
  });
  return dedupeGraphNodes(nextNodes, segs);
};

const segmentIntersect = (a1, a2, b1, b2) => {
  const x1 = a1[0], y1 = a1[1], x2 = a2[0], y2 = a2[1];
  const x3 = b1[0], y3 = b1[1], x4 = b2[0], y4 = b2[1];
  const dx12 = x2 - x1;
  const dy12 = y2 - y1;
  const dx34 = x4 - x3;
  const dy34 = y4 - y3;
  const denom = dy12 * dx34 - dx12 * dy34;
  if (Math.abs(denom) < 1e-9) return null;
  const t = ((x1 - x3) * dy34 + (y3 - y1) * dx34) / denom;
  const u = ((x3 - x1) * dy12 + (y1 - y3) * dx12) / -denom;
  if (t < 0 || t > 1 || u < 0 || u > 1) return null;
  return { x: x1 + dx12 * t, y: y1 + dy12 * t, t, u };
};

const edgeKey = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);

const splitAtIntersections = (segments) => {
  const splits = segments.map(() => [0, 1]);
  for (let i = 0; i < segments.length; i++) {
    for (let j = i + 1; j < segments.length; j++) {
      const a = segments[i];
      const b = segments[j];
      const inter = segmentIntersect(a[0], a[1], b[0], b[1]);
      if (!inter) continue;
      splits[i].push(inter.t);
      splits[j].push(inter.u);
    }
  }
  const out = [];
  for (let i = 0; i < segments.length; i++) {
    const ts = Array.from(new Set(splits[i].map((v) => Math.max(0, Math.min(1, v))))).sort((a, b) => a - b);
    const [a, b] = segments[i];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    for (let k = 0; k < ts.length - 1; k++) {
      const t0 = ts[k];
      const t1 = ts[k + 1];
      if (t1 - t0 < 1e-6) continue;
      const p0 = [a[0] + dx * t0, a[1] + dy * t0];
      const p1 = [a[0] + dx * t1, a[1] + dy * t1];
      out.push([p0, p1]);
    }
  }
  return out;
};

const snapKey = (pt, eps) => {
  if (!eps || eps <= 0) return `${Math.round(pt[0])},${Math.round(pt[1])}`;
  return `${Math.round(pt[0] / eps)},${Math.round(pt[1] / eps)}`;
};

const edgeKeyPts = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);

const buildRegionAdjacency = (polys, eps = 0.5) => {
  const edgeMap = new Map();
  polys.forEach((pts, rid) => {
    if (!pts || pts.length < 2) return;
    for (let i = 0; i < pts.length; i++) {
      const p1 = pts[i];
      const p2 = pts[(i + 1) % pts.length];
      const k1 = snapKey(p1, eps);
      const k2 = snapKey(p2, eps);
      if (k1 === k2) continue;
      const ek = edgeKeyPts(k1, k2);
      if (!edgeMap.has(ek)) edgeMap.set(ek, []);
      edgeMap.get(ek).push(rid);
    }
  });
  const adj = Array.from({ length: polys.length }, () => new Set());
  for (const ids of edgeMap.values()) {
    if (ids.length < 2) continue;
    for (let i = 0; i < ids.length; i++) {
      for (let j = i + 1; j < ids.length; j++) {
        adj[ids[i]].add(ids[j]);
        adj[ids[j]].add(ids[i]);
      }
    }
  }
  return adj.map((s) => Array.from(s));
};

const buildRegionAdjacencyMulti = (polys, epsList) => {
  const merged = Array.from({ length: polys.length }, () => new Set());
  (epsList || []).forEach((eps) => {
    const adj = buildRegionAdjacency(polys, eps);
    adj.forEach((neighbors, rid) => {
      neighbors.forEach((nb) => merged[rid].add(nb));
    });
  });
  return merged.map((s) => Array.from(s));
};

const buildZoneBoundaries = (polys, zoneId, snap = 0) => {
  const zones = new Map();
  polys.forEach((pts, rid) => {
    const zid = zoneId?.[rid] ?? -1;
    if (!zones.has(zid)) zones.set(zid, []);
    zones.get(zid).push(pts);
  });

  const zoneBoundaries = {};
  zones.forEach((zpolys, zid) => {
    const edgeCounts = new Map();
    const pointSum = new Map();

    const addPoint = (k, p) => {
      const entry = pointSum.get(k);
      if (!entry) {
        pointSum.set(k, [p[0], p[1], 1]);
      } else {
        entry[0] += p[0];
        entry[1] += p[1];
        entry[2] += 1;
      }
    };

    zpolys.forEach((pts) => {
      if (!pts || pts.length < 2) return;
      for (let i = 0; i < pts.length; i++) {
        const p1 = pts[i];
        const p2 = pts[(i + 1) % pts.length];
        const k1 = snapKey(p1, snap);
        const k2 = snapKey(p2, snap);
        if (k1 === k2) continue;
        addPoint(k1, p1);
        addPoint(k2, p2);
        const ek = edgeKeyPts(k1, k2);
        const entry = edgeCounts.get(ek);
        if (entry) {
          entry.count += 1;
        } else {
          edgeCounts.set(ek, { count: 1, a: k1, b: k2 });
        }
      }
    });

    const boundaryEdges = [];
    edgeCounts.forEach((entry) => {
      if (entry.count === 1) boundaryEdges.push([entry.a, entry.b]);
    });
    if (!boundaryEdges.length) {
      zoneBoundaries[zid] = [];
      return;
    }

    const pointMap = new Map();
    pointSum.forEach((val, key) => {
      if (val[2] > 0) pointMap.set(key, [val[0] / val[2], val[1] / val[2]]);
    });

    const adj = new Map();
    boundaryEdges.forEach(([k1, k2], idx) => {
      if (!adj.has(k1)) adj.set(k1, []);
      if (!adj.has(k2)) adj.set(k2, []);
      adj.get(k1).push(idx);
      adj.get(k2).push(idx);
    });

    const used = new Array(boundaryEdges.length).fill(false);
    const polylines = [];

    const nextEdge = (curKey, prevKey) => {
      const candidates = (adj.get(curKey) || []).filter((ei) => !used[ei]);
      if (!candidates.length) return null;
      if (!prevKey) return candidates[0];
      const pcur = pointMap.get(curKey);
      const pprev = pointMap.get(prevKey);
      if (!pcur || !pprev) return candidates[0];
      const vx = pcur[0] - pprev[0];
      const vy = pcur[1] - pprev[1];
      const vlen = Math.hypot(vx, vy);
      if (vlen === 0) return candidates[0];
      let best = candidates[0];
      let bestDot = -1e9;
      for (const ei of candidates) {
        const [a, b] = boundaryEdges[ei];
        const nxt = a === curKey ? b : a;
        const pnxt = pointMap.get(nxt);
        if (!pnxt) continue;
        const wx = pnxt[0] - pcur[0];
        const wy = pnxt[1] - pcur[1];
        const wlen = Math.hypot(wx, wy);
        if (wlen === 0) continue;
        const dot = (vx * wx + vy * wy) / (vlen * wlen);
        if (dot > bestDot) {
          bestDot = dot;
          best = ei;
        }
      }
      return best;
    };

    for (let i = 0; i < boundaryEdges.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      const [k1, k2] = boundaryEdges[i];
      const pathKeys = [k1, k2];

      while (true) {
        const cur = pathKeys[pathKeys.length - 1];
        const prev = pathKeys.length >= 2 ? pathKeys[pathKeys.length - 2] : null;
        const ei = nextEdge(cur, prev);
        if (ei == null) break;
        used[ei] = true;
        const [a, b] = boundaryEdges[ei];
        const nxt = a === cur ? b : a;
        if (nxt === pathKeys[pathKeys.length - 1]) break;
        pathKeys.push(nxt);
        if (nxt === pathKeys[0]) break;
      }

      while (true) {
        const cur = pathKeys[0];
        const prev = pathKeys.length >= 2 ? pathKeys[1] : null;
        const ei = nextEdge(cur, prev);
        if (ei == null) break;
        used[ei] = true;
        const [a, b] = boundaryEdges[ei];
        const nxt = a === cur ? b : a;
        if (nxt === pathKeys[0]) break;
        pathKeys.unshift(nxt);
        if (nxt === pathKeys[pathKeys.length - 1]) break;
      }

      const pathPts = pathKeys.map((k) => pointMap.get(k)).filter(Boolean);
      if (pathPts.length >= 2) polylines.push(pathPts);
    }

    zoneBoundaries[zid] = polylines;
  });

  return zoneBoundaries;
};

const buildSnapZoneSceneFromRegionScene = (data) => {
  if (!data?.regions?.length) return null;
  const regionCount = data.regions.length;
  const snapRegionMap = data.snap_region_map || {};
  const rawSnappedCells = data?.voronoi?.snapped_cells || [];
  const zonePolys = rawSnappedCells
    .map((poly) =>
      (poly || [])
        .map((p) =>
          Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])
            ? [Number(p[0]), Number(p[1])]
            : null
        )
        .filter(Boolean)
    )
    .filter((poly) => poly.length >= 3);
  const zoneOrder = Array.from({ length: zonePolys.length }, (_, i) => i);
  const zoneId = Array.from({ length: regionCount }, () => -1);

  Object.entries(snapRegionMap).forEach(([zidRaw, regionIds]) => {
    const zid = parseInt(zidRaw, 10);
    (regionIds || []).forEach((rid) => {
      const idx = parseInt(rid, 10);
      if (Number.isFinite(idx) && idx >= 0 && idx < regionCount) zoneId[idx] = zid;
    });
  });

  if (zonePolys.length) {
    const zoneCenters = zonePolys.map((poly) => {
      const { cx, cy } = polyCentroid(poly);
      return [cx, cy];
    });
    for (let i = 0; i < regionCount; i++) {
      if (zoneId[i] >= 0) continue;
      const { cx, cy } = polyCentroid(data.regions[i] || []);
      let bestZid = 0;
      let bestD2 = Infinity;
      zoneCenters.forEach((pt, zid) => {
        const dx = (pt?.[0] ?? 0) - cx;
        const dy = (pt?.[1] ?? 0) - cy;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
          bestD2 = d2;
          bestZid = zid;
        }
      });
      zoneId[i] = bestZid;
    }
  } else {
    for (let i = 0; i < regionCount; i++) {
      if (zoneId[i] < 0) zoneId[i] = i;
    }
  }

  const zoneBoundaries = zonePolys.length
    ? Object.fromEntries(zonePolys.map((poly, zid) => [zid, [poly]]))
    : buildZoneBoundaries(data.regions, zoneId, 0);
  const zoneLabels = {};
  const zoneLabelMap = {};
  const labelIds = zonePolys.length ? zoneOrder : Array.from(new Set(zoneId)).sort((a, b) => a - b);
  labelIds.forEach((zid, idx) => {
    if (zonePolys.length) {
      const { cx, cy } = polyCentroid(zonePolys[zid] || []);
      zoneLabels[zid] = { x: cx, y: cy, label: idx + 1 };
      zoneLabelMap[zid] = idx + 1;
      return;
    }
    let sumArea = 0;
    let sumX = 0;
    let sumY = 0;
    zoneId.forEach((ridZid, rid) => {
      if (ridZid !== zid) return;
      const poly = data.regions[rid] || [];
      const { area, cx, cy } = polyCentroid(poly);
      const w = Math.abs(area) || 1;
      sumArea += w;
      sumX += cx * w;
      sumY += cy * w;
    });
    const x = sumArea > 0 ? sumX / sumArea : 0;
    const y = sumArea > 0 ? sumY / sumArea : 0;
    zoneLabels[zid] = { x, y, label: idx + 1 };
    zoneLabelMap[zid] = idx + 1;
  });

  return {
    ...data,
    zone_id: zoneId,
    zone_boundaries: zoneBoundaries,
    zone_polys: zonePolys,
    zone_order: zoneOrder,
    zone_labels: zoneLabels,
    zone_label_map: zoneLabelMap,
  };
};

const polyAreaAndCentroid = (poly) => {
  if (!poly || poly.length < 3) return { area: 0, cx: 0, cy: 0 };
  let area = 0;
  let cx = 0;
  let cy = 0;
  const n = poly.length;
  for (let i = 0; i < n; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % n];
    const cross = x0 * y1 - x1 * y0;
    area += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  area *= 0.5;
  if (Math.abs(area) < 1e-6) return { area: 0, cx: poly[0][0], cy: poly[0][1] };
  return { area, cx: cx / (6 * area), cy: cy / (6 * area) };
};

const PACK_PRESETS = {
  fast: { label: "Fast", grid: 10, angle: 15, mode: "fast" },
  balanced: { label: "Balanced", grid: 5, angle: 5, mode: "balanced" },
  tight: { label: "Tight", grid: 2, angle: 2, mode: "tight" },
};

export default function App() {
  const sourceOnlyMode = true;
  const [snap, setSnap] = useState(1);
  const [sourceScale, setSourceScale] = useState(1);
  const [scene, setScene] = useState(null);
  const [zoneScene, setZoneScene] = useState(null);
  const [error, setError] = useState("");
  const [labels, setLabels] = useState([]);
  const [packedLabels, setPackedLabels] = useState([]);
  const [exportMsg, setExportMsg] = useState("");
  const [exportPdfInfo, setExportPdfInfo] = useState(null);
  const [exportHtmlInfo, setExportHtmlInfo] = useState([]);
  const [exportPdfLoading, setExportPdfLoading] = useState(false);
  const [exportPdfTiming, setExportPdfTiming] = useState({ startTs: 0, elapsedMs: 0 });
  const [showSim, setShowSim] = useState(false);
  const [simPlaying, setSimPlaying] = useState(false);
  const [simProgress, setSimProgress] = useState(0);
  const [simSize, setSimSize] = useState({ w: 800, h: 500 });
  const [simVideoLoading, setSimVideoLoading] = useState(false);
  const simWrapRef = useRef(null);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [rawSegments, setRawSegments] = useState([]);
  const [borderSegments, setBorderSegments] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [segs, setSegs] = useState([]);
  const [svgImage, setSvgImage] = useState(null);
  const [svgFallback, setSvgFallback] = useState([]);
  const [svgSize, setSvgSize] = useState({ w: 1000, h: 1000 });
  const [sourceVoronoi, setSourceVoronoi] = useState({ mask: [], cells: [], snappedCells: [] });
  const stageRef = useRef(null);
  const leftRef = useRef(null);
  const overlayInputRef = useRef(null);
  const overlayTransformerRef = useRef(null);
  const overlayNodeRefs = useRef({});
  const regionNeighborCursorRef = useRef({});
  const [mainViewScale, setMainViewScale] = useState(1);
  const [mainViewPos, setMainViewPos] = useState({ x: 0, y: 0 });
  const [scale, setScale] = useState(1);
  const [pos, setPos] = useState({ x: 0, y: 0 });
  const [stageSize, setStageSize] = useState({ w: 800, h: 600 });
  const regionRef = useRef(null);
  const regionWrapRef = useRef(null);
  const [regionScale, setRegionScale] = useState(1);
  const [regionPos, setRegionPos] = useState({ x: 0, y: 0 });
  const [regionStageSize, setRegionStageSize] = useState({ w: 400, h: 400 });
  const region2Ref = useRef(null);
  const region2WrapRef = useRef(null);
  const [region2Scale, setRegion2Scale] = useState(1);
  const [region2Pos, setRegion2Pos] = useState({ x: 0, y: 0 });
  const [region2StageSize, setRegion2StageSize] = useState({ w: 300, h: 200 });
  const zoneRef = useRef(null);
  const zoneWrapRef = useRef(null);
  const [zoneScale, setZoneScale] = useState(1);
  const [zonePos, setZonePos] = useState({ x: 0, y: 0 });
  const [zoneStageSize, setZoneStageSize] = useState({ w: 300, h: 200 });
  const zoneClickCacheRef = useRef([]);
  const [leftTab, setLeftTab] = useState("region");
  const rightTab = "packed";
  const neighborSnap = 0.5;
  const regionAdj = useMemo(
    () => buildRegionAdjacencyMulti((zoneScene || scene)?.regions || [], [neighborSnap, 2]),
    [zoneScene, scene]
  );
  const [autoFit, setAutoFit] = useState(true);
  const [showImages, setShowImages] = useState(true);
  const [showStroke, setShowStroke] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [labelFontFamily, setLabelFontFamily] = useState("Arial");
  const [labelFontSize, setLabelFontSize] = useState(12);
  const [packedImageSrc, setPackedImageSrc] = useState("");
  const [packedImageSrc2, setPackedImageSrc2] = useState("");
  const [packedFillPaths, setPackedFillPaths] = useState([]);
  const [packedBleedPaths, setPackedBleedPaths] = useState([]);
  const [packedBleedError, setPackedBleedError] = useState("");
  const [packedFillPaths2, setPackedFillPaths2] = useState([]);
  const [packedBleedPaths2, setPackedBleedPaths2] = useState([]);
  const [packedBleedError2, setPackedBleedError2] = useState("");
  const [packedEmptyCells, setPackedEmptyCells] = useState([]);
  const [edgeMode, setEdgeMode] = useState(false);
  const [addNodeMode, setAddNodeMode] = useState(false);
  const [deleteEdgeMode, setDeleteEdgeMode] = useState(false);
  const [edgeCandidate, setEdgeCandidate] = useState(null);
  const [deleteEdgeCandidate, setDeleteEdgeCandidate] = useState(null);
  const [sceneLoading, setSceneLoading] = useState(true);
  const [packPadding, setPackPadding] = useState(5);
  const [packMarginX, setPackMarginX] = useState(30);
  const [packMarginY, setPackMarginY] = useState(30);
  const [packBleed, setPackBleed] = useState(3);
  const [enableBleed, setEnableBleed] = useState(true);
  const [showRasterTemp, setShowRasterTemp] = useState(false);
  const [rasterTempSrc, setRasterTempSrc] = useState("");
  const [computeBusy, setComputeBusy] = useState(false);
  const [rasterBusy, setRasterBusy] = useState(false);
  const [drawScale, setDrawScale] = useState(0.5);
  const [packGrid, setPackGrid] = useState(10);
  const [packAngle, setPackAngle] = useState(15);
  const [packMode, setPackMode] = useState("fast");
  const [packPreset, setPackPreset] = useState("fast");
  const [autoPack, setAutoPack] = useState(false);
  const [overlayItems, setOverlayItems] = useState([]);
  const [selectedOverlayId, setSelectedOverlayId] = useState(null);
  const [overlayFill, setOverlayFill] = useState("#000000");
  const [zoneClickLogs, setZoneClickLogs] = useState([]);
  const [packedEditMode, setPackedEditMode] = useState("none"); // none | move | rotate
  const [manualPackedEdits, setManualPackedEdits] = useState({});
  const [packUiLog, setPackUiLog] = useState("");
  const [packTiming, setPackTiming] = useState({
    running: false,
    startTs: 0,
    elapsedMs: 0,
    lastMs: null,
    avgMs: null,
    count: 0,
  });
  const [recentFiles, setRecentFiles] = useState(["convoi.svg", "chobenthanh.svg"]);
  const [selectedSource, setSelectedSource] = useState("convoi.svg");
  const [sourceUndoStack, setSourceUndoStack] = useState([]);
  const [sourceRedoStack, setSourceRedoStack] = useState([]);
  const packedEditSessionRef = useRef(null);
  const manualPackedSaveTimerRef = useRef(null);
  const sourceDragSnapshotRef = useRef(null);
  const isPackLoading = !!packTiming.running;
  const leftPanelLoading = sceneLoading && !isPackLoading;
  const rightPanelLoading = sceneLoading;

  useEffect(() => {
    return () => {
      if (manualPackedSaveTimerRef.current) {
        clearTimeout(manualPackedSaveTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!exportPdfLoading || !exportPdfTiming.startTs) return undefined;
    const id = window.setInterval(() => {
      setExportPdfTiming((prev) => ({
        ...prev,
        elapsedMs: Math.max(0, performance.now() - prev.startTs),
      }));
    }, 100);
    return () => window.clearInterval(id);
  }, [exportPdfLoading, exportPdfTiming.startTs]);

  useEffect(() => {
    if (!packTiming.running || !packTiming.startTs) return undefined;
    const tick = () => {
      setPackTiming((prev) =>
        prev.running && prev.startTs
          ? { ...prev, elapsedMs: Math.max(0, performance.now() - prev.startTs) }
          : prev
      );
    };
    tick();
    const timer = window.setInterval(tick, 100);
    return () => window.clearInterval(timer);
  }, [packTiming.running, packTiming.startTs]);

  const handleSourceScaleChange = (value) => {
    const next = parseFloat(value);
    if (!Number.isFinite(next) || next <= 0) return;
    const prev = sourceScale || 1;
    if (Math.abs(next - prev) < 1e-6) return;
    const ratio = next / prev;
    setSourceScale(next);

    setSvgSize((s) => ({ w: s.w * ratio, h: s.h * ratio }));
    setRawSegments((s) => scaleSegments(s, ratio));
    setSvgFallback((s) => scaleSegments(s, ratio));
    setBorderSegments((s) => scaleSegments(s, ratio));
    setNodes((items) => items.map((n) => ({ ...n, x: n.x * ratio, y: n.y * ratio })));
    setOverlayItems((items) =>
      items.map((item) => ({
        ...item,
        x: item.x * ratio,
        y: item.y * ratio,
        width: item.width * ratio,
        height: item.height * ratio,
      }))
    );
    setLabels((items) =>
      items.map((lbl) => ({ ...lbl, x: lbl.x * ratio, y: lbl.y * ratio }))
    );
    setSourceVoronoi((data) => scaleVoronoiData(data, ratio));
    setPackedLabels((items) =>
      items.map((lbl) => ({ ...lbl, x: lbl.x * ratio, y: lbl.y * ratio }))
    );
    setPackedEmptyCells((cells) => cells.map((c) => [c[0] * ratio, c[1] * ratio]));

    const nextScene = scaleSceneData(scene, ratio);
    const nextZoneScene = scaleSceneData(zoneScene, ratio);
    if (nextScene !== scene) setScene(nextScene);
    if (nextZoneScene !== zoneScene) setZoneScene(nextZoneScene);

    if (zoneClickCacheRef.current?.length) {
      zoneClickCacheRef.current = zoneClickCacheRef.current.map((pt) => ({
        ...pt,
        x: pt.x * ratio,
        y: pt.y * ratio,
      }));
      fetch(`/api/source_zone_click?${sourceQuery(selectedSource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clicks: zoneClickCacheRef.current }),
      }).catch(() => {});
    }

    const packedSource = nextZoneScene || nextScene;
    if (!sourceOnlyMode && leftTab !== "source" && packedSource) {
      const packedPolyData = buildPackedPolyData(packedSource);
      setPackedEmptyCells(buildPackedEmptyCells(packedSource, packedPolyData));
      refreshPackedFromZoneScene(packedSource, enableBleed, selectedSource);
    }
  };

  const sourceVoronoiGraph = useMemo(
    () => buildVoronoiVertexGraph(sourceVoronoi?.snappedCells || []),
    [sourceVoronoi]
  );
  const liveSnapRadius = Math.max(0, (snap || 0) * 0.35);
  const greenToRedSnapRadius = 50;

  const createSourceEditSnapshot = (
    nextNodes = nodes,
    nextSegs = segs,
    nextVoronoi = sourceVoronoi
  ) => ({
    nodes: cloneSourceNodes(nextNodes),
    segs: cloneSourceSegs(nextSegs),
    voronoi: cloneSourceVoronoi(nextVoronoi),
  });

  const sourceSnapshotsEqual = (a, b) =>
    JSON.stringify(a || null) === JSON.stringify(b || null);

  const applySourceEditSnapshot = (snapshot) => {
    if (!snapshot) return;
    setNodes(cloneSourceNodes(snapshot.nodes));
    setSegs(cloneSourceSegs(snapshot.segs));
    setSourceVoronoi(cloneSourceVoronoi(snapshot.voronoi));
  };

  const pushSourceUndoSnapshot = (snapshot) => {
    if (!snapshot) return;
    setSourceUndoStack((prev) => {
      const next = [...prev, snapshot];
      return next.length > 100 ? next.slice(next.length - 100) : next;
    });
    setSourceRedoStack([]);
  };

  const commitSourceDragHistory = (
    nextNodes = nodes,
    nextSegs = segs,
    nextVoronoi = sourceVoronoi
  ) => {
    const before = sourceDragSnapshotRef.current;
    sourceDragSnapshotRef.current = null;
    if (!before) return;
    const after = createSourceEditSnapshot(nextNodes, nextSegs, nextVoronoi);
    if (!sourceSnapshotsEqual(before, after)) {
      pushSourceUndoSnapshot(before);
    }
  };

  const resetSourceHistory = () => {
    sourceDragSnapshotRef.current = null;
    setSourceUndoStack([]);
    setSourceRedoStack([]);
  };

  const undoSourceEdit = () => {
    if (!sourceUndoStack.length) return;
    const current = createSourceEditSnapshot();
    const snapshot = sourceUndoStack[sourceUndoStack.length - 1];
    setSourceUndoStack((prev) => prev.slice(0, -1));
    setSourceRedoStack((prev) => {
      const next = [...prev, current];
      return next.length > 100 ? next.slice(next.length - 100) : next;
    });
    applySourceEditSnapshot(snapshot);
  };

  const redoSourceEdit = () => {
    if (!sourceRedoStack.length) return;
    const current = createSourceEditSnapshot();
    const snapshot = sourceRedoStack[sourceRedoStack.length - 1];
    setSourceRedoStack((prev) => prev.slice(0, -1));
    setSourceUndoStack((prev) => {
      const next = [...prev, current];
      return next.length > 100 ? next.slice(next.length - 100) : next;
    });
    applySourceEditSnapshot(snapshot);
  };

  const resolveSourceSnapTarget = (x, y, options = {}) => {
    const {
      excludeNodeId = null,
      excludeVoronoiId = null,
      includeNodes = true,
      includeVoronoi = true,
      radius = liveSnapRadius,
    } = options;
    if (!(radius > 0)) return { x, y, kind: "none", id: null };
    let best = null;
    let bestD2 = radius * radius;
    if (includeNodes) {
      (nodes || []).forEach((n) => {
        if (excludeNodeId != null && n.id === excludeNodeId) return;
        const dx = n.x - x;
        const dy = n.y - y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= bestD2) {
          bestD2 = d2;
          best = { x: n.x, y: n.y, kind: "node", id: n.id };
        }
      });
    }
    if (includeVoronoi) {
      (sourceVoronoiGraph?.vertices || []).forEach((v) => {
        if (excludeVoronoiId != null && v.id === excludeVoronoiId) return;
        const dx = v.x - x;
        const dy = v.y - y;
        const d2 = dx * dx + dy * dy;
        if (d2 <= bestD2) {
          bestD2 = d2;
          best = { x: v.x, y: v.y, kind: "voronoi", id: v.id };
        }
      });
    }
    return best || { x, y, kind: "none", id: null };
  };

  const updateSourceVoronoiVertex = (vertexId, x, y, options = {}) => {
    const { snapTarget = null } = options;
    const target = snapTarget || { x, y, kind: "none", id: null };
    const currentVoronoi = sourceVoronoi || { mask: [], cells: [], snappedCells: [] };
    const graph = buildVoronoiVertexGraph(currentVoronoi?.snappedCells || []);
    if (!graph.vertices[vertexId]) return target;
    let nextVertices = graph.vertices.map((v) =>
      v.id === vertexId ? { ...v, x: target.x, y: target.y } : { ...v }
    );
    const nextVoronoi = {
      ...currentVoronoi,
      snappedCells: rebuildVoronoiCellsFromGraph(nextVertices, graph.refs),
    };
    setSourceVoronoi(nextVoronoi);
    return target;
  };

  const applyPackPreset = (name, rerender = false) => {
    const key = String(name || "").toLowerCase();
    const preset = PACK_PRESETS[key];
    if (!preset) return;
    clearManualPackedEdits();
    setPackPreset(key);
    setPackGrid(preset.grid);
    setPackAngle(preset.angle);
    setPackMode(preset.mode);
    if (rerender) {
      requestAnimationFrame(() => loadScene(false, true, true));
    }
  };

  const simZoneIds = useMemo(() => {
    const ids = Object.keys(scene?.zone_boundaries || {});
    const getLabel = (zid) => {
      const lbl =
        scene?.zone_label_map?.[zid] ??
        scene?.zone_label_map?.[parseInt(zid, 10)] ??
        zid;
      const num = Number(lbl);
      return Number.isFinite(num) ? num : Number(zid) || 0;
    };
    return ids.sort((a, b) => getLabel(a) - getLabel(b));
  }, [scene]);
  const simZoneIndex = useMemo(() => {
    const map = {};
    simZoneIds.forEach((zid, idx) => {
      map[String(zid)] = idx;
    });
    return map;
  }, [simZoneIds]);
  const simTiming = useMemo(() => {
    const move = 1;
    const hold = 0.2;
    const per = move + hold;
    const total = simZoneIds.length ? simZoneIds.length * per : 1;
    return { move, hold, per, total };
  }, [simZoneIds]);
  const simMoveSeconds = simTiming.move;
  const simHoldSeconds = simTiming.hold;
  const simPerZone = simTiming.per;
  const simTotalSeconds = simTiming.total;
  const simActiveIdx = simZoneIds.length
    ? Math.min(
        simZoneIds.length - 1,
        Math.max(0, Math.floor((simProgress * simTotalSeconds) / simPerZone))
      )
    : -1;
  const simActiveZid = simActiveIdx >= 0 ? simZoneIds[simActiveIdx] : null;
  const simActiveLabel =
    simActiveZid != null
      ? scene?.zone_label_map?.[simActiveZid] ??
        scene?.zone_label_map?.[parseInt(simActiveZid, 10)] ??
        simActiveZid
      : "";
  const zoneLabelCenters = useMemo(() => {
    const source = zoneScene || scene;
    const centers = {};
    const boundaries = source?.zone_boundaries || {};
    Object.entries(boundaries).forEach(([zid, paths]) => {
      if (!paths || !paths.length) return;
      let best = null;
      for (const p of paths) {
        if (!p || p.length < 3) continue;
        const { area, cx, cy } = polyAreaAndCentroid(p);
        const absArea = Math.abs(area);
        if (!best || absArea > best.absArea) {
          best = { absArea, cx, cy };
        }
      }
      if (best && Number.isFinite(best.cx) && Number.isFinite(best.cy)) {
        centers[String(zid)] = { x: best.cx, y: best.cy };
      }
    });
    return centers;
  }, [zoneScene, scene]);
  const zoneAreaStats = useMemo(() => {
    const source = zoneScene || scene;
    const boundaries = source?.zone_boundaries || {};
    const zoneAreas = [];
    Object.values(boundaries).forEach((paths) => {
      let zoneArea = 0;
      (paths || []).forEach((p) => {
        const { area } = polyAreaAndCentroid(p);
        zoneArea += Math.abs(area || 0);
      });
      if (zoneArea > 0) zoneAreas.push(zoneArea);
    });
    const count = zoneAreas.length;
    const avg = count ? zoneAreas.reduce((a, b) => a + b, 0) / count : 0;
    return { count, avg };
  }, [zoneScene, scene]);
  const simLocalFor = (idx) => {
    if (idx == null || idx < 0) return 0;
    const t = simProgress * simTotalSeconds - idx * simPerZone;
    if (t <= 0) return 0;
    if (t >= simPerZone) return 1;
    if (t >= simMoveSeconds) return 1;
    const x = t / simMoveSeconds;
    return 1 - Math.pow(1 - x, 3);
  };

  const simStage = scene?.canvas
    ? (() => {
        const gap = 20;
        const totalW = (scene.canvas.w * 2) + gap;
        const totalH = scene.canvas.h;
        const fitScale = Math.min(simSize.w / totalW, simSize.h / totalH) * 1.06;
        const offsetX = (simSize.w - totalW * fitScale) / 2;
        const offsetY = (simSize.h - totalH * fitScale) / 2;
        return (
          <Stage
            width={simSize.w}
            height={simSize.h}
            scaleX={fitScale}
            scaleY={fitScale}
            x={offsetX}
            y={offsetY}
          ><Layer><Rect
                x={0}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={1}
              /><Rect
                x={scene.canvas.w + gap}
                y={0}
                width={scene.canvas.w}
                height={scene.canvas.h}
                stroke="#ffffff"
                strokeWidth={1}
              /></Layer><Layer>{scene.region_colors
                ? scene.regions.map((poly, idx) => {
                    const zid = scene.zone_id?.[idx];
                    const zidKey = String(zid);
                    const zoneIdx = simZoneIndex[zidKey] ?? 0;
                    const local = simLocalFor(zoneIdx);
                    if (local > 0) return null;
                    const shift =
                      scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                    if (!shift) return null;
                    const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                    const center =
                      scene.zone_center?.[zid] || scene.zone_center?.[parseInt(zid, 10)] || [0, 0];
                    const tpts = transformPath(poly, shift, rot, center);
                    return (
                      <Line
                        key={`sim-pack-fill-${idx}`}
                        points={toPoints(tpts)}
                        closed
                        fill={scene.region_colors[idx]}
                        strokeScaleEnabled={false}
                      />
                    );
                  })
                : null}{packedLabels.map((lbl) => {
                const zidKey = String(lbl.zid);
                const zoneIdx = simZoneIndex[zidKey] ?? 0;
                const local = simLocalFor(zoneIdx);
                if (local > 0) return null;
                const size = Math.max(labelFontSize * 0.5, 6);
                const metrics = measureText(lbl.label, size, labelFontFamily);
                return (
                  <Text
                    key={`sim-pack-label-${lbl.id}`}
                    x={lbl.x}
                    y={lbl.y}
                    text={lbl.label}
                    fill="#ffffff"
                    fontSize={size}
                    fontFamily={labelFontFamily}
                    align="center"
                    verticalAlign="middle"
                    offsetX={metrics.width / 2}
                    offsetY={metrics.height / 2}
                  />
                );
              })}</Layer><Layer>{Object.values(scene.zone_labels || {}).map((lbl) => {
                const size = Math.max(labelFontSize * 0.5, 6);
                const metrics = measureText(lbl.label, size, labelFontFamily);
                return (
                  <Text
                    key={`sim-zone-label-${lbl.label}`}
                    x={lbl.x + scene.canvas.w + gap}
                    y={lbl.y}
                    text={lbl.label}
                    fill="#ffffff"
                    fontSize={size}
                    fontFamily={labelFontFamily}
                    align="center"
                    verticalAlign="middle"
                    offsetX={metrics.width / 2}
                    offsetY={metrics.height / 2}
                  />
                );
              })}</Layer><Layer>{simZoneIds.flatMap((zid) => {
                const paths = scene.zone_boundaries?.[zid] || [];
                return paths.map((p, i) => (
                  <Line
                    key={`sim-zone-${zid}-${i}`}
                    points={toPoints(offsetPoints(p, scene.canvas.w + gap, 0))}
                    stroke="#ffffff"
                    strokeWidth={1}
                    closed
                  />
                ));
              })}</Layer><Layer>{scene.region_colors
                ? scene.regions.map((poly, idx) => {
                    const zid = scene.zone_id?.[idx];
                    const zidKey = String(zid);
                    const zoneIdx = simZoneIndex[zidKey] ?? 0;
                    const local = simLocalFor(zoneIdx);
                    const shift =
                      scene.zone_shift?.[zid] || scene.zone_shift?.[parseInt(zid, 10)];
                    if (!shift) return null;
                    const rot = scene.zone_rot?.[zid] ?? scene.zone_rot?.[parseInt(zid, 10)] ?? 0;
                    const center =
                      scene.zone_center?.[zid] ||
                      scene.zone_center?.[parseInt(zid, 10)] ||
                      [0, 0];
                    const src = transformPath(poly, shift, rot, center);
                    const dst = offsetPoints(poly, scene.canvas.w + gap, 0);
                    const pts =
                      local >= 1
                        ? dst
                        : src.map((sp, k) => {
                            const dp = dst[k] || sp;
                            return [lerp(sp[0], dp[0], local), lerp(sp[1], dp[1], local)];
                          });
                    if (local <= 0) return null;
                    return (
                      <Line
                        key={`sim-move-fill-${idx}`}
                        points={toPoints(pts)}
                        closed
                        fill={scene.region_colors[idx]}
                        strokeScaleEnabled={false}
                      />
                    );
                  })
                : null}</Layer></Stage>
        );
      })()
    : null;

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/recent_sources");
        if (res.ok) {
          const data = await res.json();
          const files = Array.isArray(data?.files) && data.files.length ? data.files : ["convoi.svg"];
          const active = data?.active && files.includes(data.active) ? data.active : files[0];
          setRecentFiles(files);
          setSelectedSource(active);
          await loadScene(true, false, false, false, active);
          return;
        }
      } catch {}
      await loadScene(true, false, false, false, "convoi.svg");
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const updateSize = () => {
      if (!leftRef.current) return;
      const rect = leftRef.current.getBoundingClientRect();
      setStageSize({ w: Math.max(300, rect.width), h: Math.max(300, rect.height) });
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  useEffect(() => {
    const updateRegionSize = () => {
      if (!regionWrapRef.current) return;
      const rect = regionWrapRef.current.getBoundingClientRect();
      setRegionStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegionSize();
    window.addEventListener("resize", updateRegionSize);
    return () => window.removeEventListener("resize", updateRegionSize);
  }, []);

  useEffect(() => {
    const updateRegion2Size = () => {
      if (!region2WrapRef.current) return;
      const rect = region2WrapRef.current.getBoundingClientRect();
      setRegion2StageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateRegion2Size();
    window.addEventListener("resize", updateRegion2Size);
    return () => window.removeEventListener("resize", updateRegion2Size);
  }, []);

  useEffect(() => {
    const updateZoneSize = () => {
      if (!zoneWrapRef.current) return;
      const rect = zoneWrapRef.current.getBoundingClientRect();
      setZoneStageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
    };
    updateZoneSize();
    window.addEventListener("resize", updateZoneSize);
    return () => window.removeEventListener("resize", updateZoneSize);
  }, []);

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      if (leftTab === "source" && leftRef.current) {
        const rect = leftRef.current.getBoundingClientRect();
        setStageSize({ w: Math.max(300, rect.width), h: Math.max(300, rect.height) });
      }
      if (leftTab === "region" && region2WrapRef.current) {
        const rect = region2WrapRef.current.getBoundingClientRect();
        setRegion2StageSize({ w: Math.max(200, rect.width), h: Math.max(200, rect.height) });
      }
    });
    return () => cancelAnimationFrame(raf);
  }, [leftTab]);

  useEffect(() => {
    const updateSimSize = () => {
      if (!simWrapRef.current) return;
      const rect = simWrapRef.current.getBoundingClientRect();
      setSimSize({ w: Math.max(300, rect.width), h: Math.max(200, rect.height) });
    };
    updateSimSize();
    window.addEventListener("resize", updateSimSize);
    return () => window.removeEventListener("resize", updateSimSize);
  }, []);

  const fitMainViewToView = (bounds) => {
    let viewW, viewH;
    if (leftTab === "source") {
      const rect = leftRef.current?.getBoundingClientRect();
      viewW = rect?.width || stageSize.w;
      viewH = rect?.height || stageSize.h;
    } else {
      const rect = region2WrapRef.current?.getBoundingClientRect();
      viewW = rect?.width || region2StageSize.w;
      viewH = rect?.height || region2StageSize.h;
    }

    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    if (!w || !h || !viewW || !viewH) return;

    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setMainViewScale(fitScale);
    setMainViewPos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitToView = (w, h) => {
    const viewW = stageSize.w;
    const viewH = stageSize.h;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setScale(fitScale);
    setPos({
      x: (viewW - w * fitScale) / 2,
      y: (viewH - h * fitScale) / 2,
    });
  };

  const fitRegionToView = (bounds) => {
    const rect = regionWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || regionStageSize.w;
    const viewH = rect?.height || regionStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegionScale(fitScale);
    setRegionPos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitRegion2ToView = (bounds) => {
    const rect = region2WrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || region2StageSize.w;
    const viewH = rect?.height || region2StageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setRegion2Scale(fitScale);
    setRegion2Pos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  const fitZoneToView = (bounds) => {
    const rect = zoneWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || zoneStageSize.w;
    const viewH = rect?.height || zoneStageSize.h;
    const w = bounds.maxx - bounds.minx;
    const h = bounds.maxy - bounds.miny;
    const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
    setZoneScale(fitScale);
    setZonePos({
      x: (viewW - w * fitScale) / 2 - bounds.minx * fitScale,
      y: (viewH - h * fitScale) / 2 - bounds.miny * fitScale,
    });
  };

  useEffect(() => {
    if (autoFit && svgSize.w && svgSize.h) {
      fitToView(svgSize.w, svgSize.h);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [svgSize, stageSize, autoFit]);

  useEffect(() => {
    if (!scene || !overlayItems.length) return;
    const updated = overlayItems.map((item) => {
      if (!item) return item;
      const zid = findZoneAtPoint({ x: item.x, y: item.y });
      return { ...item, zid: zid ?? item.zid ?? null };
    });
    const changed = updated.some((item, idx) => item?.zid !== overlayItems[idx]?.zid);
    if (changed) setOverlayItems(updated);
  }, [scene]);

  useEffect(() => {
    const tr = overlayTransformerRef.current;
    if (!tr) return;
    const node = selectedOverlayId ? overlayNodeRefs.current[selectedOverlayId] : null;
    if (node) {
      tr.nodes([node]);
    } else {
      tr.nodes([]);
    }
    tr.getLayer()?.batchDraw?.();
  }, [selectedOverlayId, overlayItems]);

  useEffect(() => {
    const onKey = (e) => {
      if (!selectedOverlayId) return;
      if (e.key !== "Delete" && e.key !== "Backspace") return;
        setOverlayItems((items) => items.filter((item) => item.id !== selectedOverlayId));
        setSelectedOverlayId(null);
      };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedOverlayId, nodes, segs, overlayItems]);

  const sourceOutPath = (name) => {
    const safe = String(name || "convoi.svg").replace(/^\/+/, "");
    const base = safe === "convoi.svg" ? "/out/convoi.svg" : `/out/sources/${encodeURIComponent(safe)}`;
    return `${base}?t=${Date.now()}`;
  };

  const sourceQuery = useCallback(
    (name) => `source=${encodeURIComponent(name || selectedSource || "convoi.svg")}`,
    [selectedSource]
  );

  const [zoneVertices, setZoneVertices] = useState({});

  useEffect(() => {
    let active = true;
    if (!selectedSource) {
      setZoneVertices({});
      return;
    }
    const query = sourceQuery(selectedSource);
    fetch(`/api/zone_debug?${query}`)
      .then((res) => res.json())
      .then((data) => {
        if (!active) return;
        setZoneVertices(data.vertices || {});
      })
      .catch(() => {
        if (!active) return;
        setZoneVertices({});
      });
    return () => {
      active = false;
    };
  }, [selectedSource, sourceQuery]);

  const loadScene = async (
    fit = true,
    updatePacked = true,
    updateZone = true,
    forceCompute = false,
    sourceOverride = null
  ) => {
    try {
      const sourceName = sourceOverride || selectedSource || "convoi.svg";
      const sq = sourceQuery(sourceName);
      setError("");
      setAutoFit(fit);
      setSceneLoading(true);
      let savedView = null;
      let savedState = {};
      try {
        const stateRes = await fetch(`/api/state?${sq}`);
        if (stateRes.ok) {
          const stateJson = await stateRes.json();
          savedView = stateJson?.view || null;
          savedState = stateJson || {};
        }
      } catch {
        savedView = null;
        savedState = {};
      }
      const svgRes = await fetch(sourceOutPath(sourceName));
      if (!svgRes.ok) throw new Error(`svg fetch failed: ${svgRes.status}`);
      const svgText = await svgRes.text();
      const scaleRatio = sourceScale || 1;
      const parsedSize = parseSvgSize(svgText);
      const scaledSize =
        scaleRatio === 1 ? parsedSize : { w: parsedSize.w * scaleRatio, h: parsedSize.h * scaleRatio };
      setSvgSize(scaledSize);
      const overlayParsed = parseOverlayItems(svgText);
      if (overlayParsed.length) {
        const hydrated = (
          await Promise.all(
            overlayParsed.map(async (item) => {
              const img = await loadImageFromSrc(item.src);
              const width = item.width || img?.width || 0;
              const height = item.height || img?.height || 0;
              return {
                ...item,
                width,
                height,
                scaleX: Number.isFinite(item.scaleX) ? item.scaleX : 1,
                scaleY: Number.isFinite(item.scaleY) ? item.scaleY : 1,
                rotation: Number.isFinite(item.rotation) ? item.rotation : 0,
                img,
              };
            })
          )
        ).filter(Boolean);
        const scaledOverlays =
          scaleRatio === 1
            ? hydrated
            : hydrated.map((item) => ({
                ...item,
                x: item.x * scaleRatio,
                y: item.y * scaleRatio,
                width: item.width * scaleRatio,
                height: item.height * scaleRatio,
              }));
        setOverlayItems(scaledOverlays);
      } else {
        setOverlayItems([]);
      }
      setSelectedOverlayId(null);
      const parsed = buildSegmentsFromSvg(svgText);
      const segmentsRaw = parsed.segments;
      const bordersRaw = parsed.borderSegments;
      const segments = scaleRatio === 1 ? segmentsRaw : scaleSegments(segmentsRaw, scaleRatio);
      const borders = scaleRatio === 1 ? bordersRaw : scaleSegments(bordersRaw, scaleRatio);
      setSvgFallback(segments);
      setBorderSegments(borders);
      // no background rendering; keep only geometry
      setRawSegments(segments);
      const cachedNodesRaw = Array.isArray(savedState?.svg_nodes) ? savedState.svg_nodes : null;
      const cachedSegsRaw = Array.isArray(savedState?.svg_segments) ? savedState.svg_segments : null;
      const cachedVoronoiRaw = savedState?.source_voronoi && typeof savedState.source_voronoi === "object"
        ? savedState.source_voronoi
        : null;
      let nextGraph = null;
      if (cachedNodesRaw?.length && cachedSegsRaw?.length) {
        nextGraph = {
          nodes: cachedNodesRaw.map((n, idx) => ({
            id: idx,
            x: (Number(n?.x) || 0) * scaleRatio,
            y: (Number(n?.y) || 0) * scaleRatio,
          })),
          segs: cachedSegsRaw
            .map((seg) => [parseInt(seg?.[0], 10), parseInt(seg?.[1], 10)])
            .filter(([a, b]) => Number.isFinite(a) && Number.isFinite(b)),
        };
      } else {
        const nonBorder = segments.filter(
          (seg) =>
            !borders.some(
              (b) =>
                b[0][0] === seg[0][0] &&
                b[0][1] === seg[0][1] &&
                b[1][0] === seg[1][0] &&
                b[1][1] === seg[1][1]
            )
        );
        const splitSegments = splitAtIntersections(nonBorder);
        nextGraph = snapNodes(splitSegments, snap);
      }
      if (cachedVoronoiRaw) {
        const scaledVoronoi = scaleVoronoiData(
          {
            ...cachedVoronoiRaw,
            snappedCells: cachedVoronoiRaw.snappedCells || cachedVoronoiRaw.snapped_cells || [],
          },
          scaleRatio
        );
        setSourceVoronoi({
          mask: scaledVoronoi.mask || [],
          cells: scaledVoronoi.cells || [],
          snappedCells: scaledVoronoi.snappedCells || [],
        });
      } else {
        try {
          const voronoiRes = await fetch(`/api/source_voronoi?${sq}&count=120&relax=2&seed=7`);
          if (!voronoiRes.ok) throw new Error(`source voronoi fetch failed: ${voronoiRes.status}`);
          const voronoiData = await voronoiRes.json();
          const scaledVoronoi = scaleVoronoiData(
            {
              ...voronoiData,
              snappedCells: voronoiData.snapped_cells || [],
            },
            scaleRatio
          );
          setSourceVoronoi({
            mask: scaledVoronoi.mask || [],
            cells: scaledVoronoi.cells || [],
            snappedCells: scaledVoronoi.snappedCells || [],
          });
        } catch {
          setSourceVoronoi({ mask: [], cells: [], snappedCells: [] });
        }
      }
      setNodes(nextGraph.nodes);
      setSegs(nextGraph.segs);
      resetSourceHistory();

      if (!updatePacked && !updateZone) {
        try {
          const regionRes = await fetch(
            `/api/source_region_scene?${sq}&count=120&relax=2&seed=7&_ts=${Date.now()}`,
            { cache: "no-store" }
          );
          if (!regionRes.ok) throw new Error(`source region scene fetch failed: ${regionRes.status}`);
          const regionData = await regionRes.json();
          const scaledRegionData = scaleRatio === 1 ? regionData : scaleSceneData(regionData, scaleRatio);
          setScene(scaledRegionData);
        } catch {
          setScene((prev) => ({
            ...(prev || {}),
            canvas: { w: scaledSize.w, h: scaledSize.h },
            regions: [],
            source_name: sourceName,
          }));
        }
        if (savedView?.source?.scale && savedView?.source?.pos) {
          setScale(savedView.source.scale);
          setPos(savedView.source.pos);
          setAutoFit(false);
        }
        if (fit && !(savedView?.source?.scale && savedView?.source?.pos)) {
          const w = parsedSize.w || 1200;
          const h = parsedSize.h || 800;
          const viewW = stageSize.w;
          const viewH = stageSize.h;
          const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
          setScale(fitScale);
          setPos({
            x: (viewW - w * fitScale) / 2,
            y: (viewH - h * fitScale) / 2,
          });
        }
        setSceneLoading(false);
        setSelectedSource(sourceName);
        return;
      }

      const res = await fetch(
        `/api/scene?${sq}&snap=${snap}&pack_padding=${packPadding}&pack_margin_x=${packMarginX}&pack_margin_y=${packMarginY}&pack_bleed=${packBleed}&draw_scale=${drawScale}&pack_grid=${packGrid}&pack_angle=${packAngle}&pack_mode=${packMode}&force_compute=${
          1
        }`
      );
      if (!res.ok) {
        throw new Error(`scene fetch failed: ${res.status}`);
      }
      const data = await res.json();
      const scaledData = scaleRatio === 1 ? data : scaleSceneData(data, scaleRatio);
      setScene(scaledData);
      if (updateZone) {
        const zoneData = scaledData;
        zoneClickCacheRef.current = [];
        setManualPackedEdits({});
        setZoneScene(zoneData);
        refreshPackedFromZoneScene(zoneData, enableBleed, sourceName).catch(() => {});
      }
      logPackedPreview(scaledData);
      if (typeof scaledData.draw_scale === "number") {
        setDrawScale(scaledData.draw_scale);
      }
        const initLabels = Object.values(scaledData.zone_labels || {}).map((v) => ({
          id: `z-${v.label}`,
          x: v.x,
          y: v.y,
          label: `${v.label}`,
        }));
        setLabels(initLabels);
        if (updatePacked) {
          const packedPolyData = buildPackedPolyData(scaledData);
          const emptyCells = buildPackedEmptyCells(scaledData, packedPolyData);
          setPackedEmptyCells(emptyCells);
          let cachedPacked = {};
          try {
            const labelRes = await fetch(`/api/packed_labels?${sq}`);
            if (labelRes.ok) {
              cachedPacked = (await labelRes.json()) || {};
            }
          } catch {
            cachedPacked = {};
          }
          const usedCell = new Set();
          const cellIndex = (pt) => `${Math.round(pt[0] / 10)}:${Math.round(pt[1] / 10)}`;
          const nextPackedLabels = Object.entries(scaledData.zone_labels || {}).map(([zid, v]) => {
        const shift = scaledData.zone_shift?.[zid] || scaledData.zone_shift?.[parseInt(zid, 10)];
        const rot = scaledData.zone_rot?.[zid] ?? scaledData.zone_rot?.[parseInt(zid, 10)] ?? 0;
        const center = scaledData.zone_center?.[zid] || scaledData.zone_center?.[parseInt(zid, 10)] || [0, 0];
        let px = v.x;
        let py = v.y;
        const cached = cachedPacked?.[String(zid)];
        if (cached && Number.isFinite(cached.x) && Number.isFinite(cached.y)) {
          px = cached.x;
          py = cached.y;
        } else {
          let tx = v.x;
          let ty = v.y;
          if (shift) {
            const [pt] = transformPath([[v.x, v.y]], shift, rot, center);
            if (pt) {
              tx = pt[0];
              ty = pt[1];
            }
          }
          let best = null;
          let bestScore = Infinity;
          const lx = Math.round(tx / 10);
          const ly = Math.round(ty / 10);
          const minCx = lx - 10;
          const maxCx = lx + 10;
          const minCy = ly - 10;
          const maxCy = ly + 10;
          for (const cell of emptyCells) {
            const idx = cellIndex(cell);
            if (usedCell.has(idx)) continue;
            const cx = Math.round(cell[0] / 10);
            const cy = Math.round(cell[1] / 10);
            if (cx < minCx || cx > maxCx || cy < minCy || cy > maxCy) continue;
            const score = Math.abs(cx - lx) + Math.abs(cy - ly);
            if (score < bestScore) {
              bestScore = score;
              best = cell;
            }
          }
          if (best) {
            px = best[0];
            py = best[1];
            usedCell.add(cellIndex(best));
          } else {
            px = tx;
            py = ty;
          }
        }
        if (data.canvas) {
          const r = 3;
          const maxX = data.canvas.w - r;
          const maxY = data.canvas.h - r;
          px = Math.max(r, Math.min(maxX, px));
          py = Math.max(r, Math.min(maxY, py));
        }
        const mapped = data.zone_label_map?.[zid] ?? data.zone_label_map?.[parseInt(zid, 10)];
        const label = mapped != null ? mapped : v.label;
            return { id: `pz-${zid}`, zid: String(zid), x: px, y: py, label: `${label}` };
          });
          setPackedLabels(nextPackedLabels);
        }
        if (savedView?.source?.scale && savedView?.source?.pos) {
          setScale(savedView.source.scale);
          setPos(savedView.source.pos);
          setAutoFit(false);
        }
        if (savedView?.region?.scale && savedView?.region?.pos) {
          setRegionScale(savedView.region.scale);
          setRegionPos(savedView.region.pos);
        }
        if (savedView?.region2?.scale && savedView?.region2?.pos) {
          setRegion2Scale(savedView.region2.scale);
          setRegion2Pos(savedView.region2.pos);
        }
        if (savedView?.zone?.scale && savedView?.zone?.pos) {
          setZoneScale(savedView.zone.scale);
          setZonePos(savedView.zone.pos);
        }
        if (fit && !(savedView?.source?.scale && savedView?.source?.pos)) {
          const w = parsedSize.w || data.canvas?.w || 1200;
          const h = parsedSize.h || data.canvas?.h || 800;
          const viewW = stageSize.w;
          const viewH = stageSize.h;
        const fitScale = Math.min(viewW / w, viewH / h) * 0.95;
        setScale(fitScale);
        setPos({
          x: (viewW - w * fitScale) / 2,
          y: (viewH - h * fitScale) / 2,
        });
        if (data.canvas) {
          fitRegionToView({ minx: 0, miny: 0, maxx: data.canvas.w, maxy: data.canvas.h });
        } else {
          fitRegionToView(calcBounds(data.regions || []));
        }
        const regionBounds = calcBounds(data.regions || []);
        fitRegion2ToView(regionBounds);
        const zoneBounds = calcBoundsFromLines(data.zone_boundaries);
        fitZoneToView(zoneBounds);
      }
      setSceneLoading(false);
      setSelectedSource(sourceName);
    } catch (err) {
      setError(err.message || String(err));
      setSceneLoading(false);
    }
  };

  const serializeOverlays = (items) =>
    (items || []).map((item) => ({
      id: item.id,
      src: item.src,
      x: item.x,
      y: item.y,
      width: item.width,
      height: item.height,
      scaleX: item.scaleX,
      scaleY: item.scaleY,
      rotation: item.rotation,
      zid: item.zid ?? null,
    }));

  const saveSvg = (nextNodes = nodes, nextSegs = segs, nextOverlays = overlayItems) =>
    fetch(`/api/save_svg?${sourceQuery(selectedSource)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: selectedSource,
        nodes: nextNodes,
        segs: nextSegs,
        overlays: serializeOverlays(nextOverlays),
      }),
    });

  const updateOverlayItem = (id, patch) => {
    setOverlayItems((items) =>
      items.map((item) => (item.id === id ? { ...item, ...patch } : item))
    );
  };

  const handleOverlayPick = () => {
    overlayInputRef.current?.click();
  };

  const handleOverlayFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const raw = await file.text();
    const filled = applySvgFill(raw, overlayFill);
    const src = svgToDataUrl(filled);
    const img = await loadImageFromSrc(src);
    const width = img?.width || parseSvgSize(raw).w || 1;
    const height = img?.height || parseSvgSize(raw).h || 1;
    const id = `overlay-${Date.now()}`;
    const item = {
      id,
      src,
      rawSvg: raw,
      x: svgSize.w * 0.5,
      y: svgSize.h * 0.5,
      width,
      height,
      scaleX: 1,
      scaleY: 1,
      rotation: 0,
      img,
      zid: findZoneAtPoint({ x: svgSize.w * 0.5, y: svgSize.h * 0.5 }),
    };
    const next = [...overlayItems, item];
    setOverlayItems(next);
    setSelectedOverlayId(id);
    e.target.value = "";
  };

  const updateOverlayColor = async (id, color) => {
    const item = overlayItems.find((i) => i.id === id);
    if (!item) return;
    const raw = item.rawSvg || decodeSvgDataUrl(item.src);
    if (!raw) return;
    const filled = applySvgFill(raw, color);
    const src = svgToDataUrl(filled);
    const img = await loadImageFromSrc(src);
    const next = overlayItems.map((i) =>
      i.id === id ? { ...i, src, img, rawSvg: raw } : i
    );
    setOverlayItems(next);
  };

  const buildSimulateHtml = (data, packed, fontFamily, fontSize) => {
    if (!data?.canvas) return "";
    const payload = {
      canvas: data.canvas,
      regions: data.regions || [],
      zone_id: data.zone_id || [],
      zone_shift: data.zone_shift || {},
      zone_rot: data.zone_rot || {},
      zone_center: data.zone_center || {},
      zone_labels: data.zone_labels || {},
      zone_boundaries: data.zone_boundaries || {},
      zone_label_map: data.zone_label_map || {},
      region_colors: data.region_colors || [],
      packed_labels: packed || [],
      font_family: fontFamily || "Arial",
      font_size: fontSize || 12,
    };
    return `<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Simulate</title>
    <style>
      html, body { margin: 0; padding: 0; background: #0b1022; color: #e8ebff; font-family: Arial, sans-serif; }
      .wrap { width: 100vw; height: 100vh; display: flex; flex-direction: column; }
      .topbar { height: 48px; display: flex; align-items: center; justify-content: center; position: relative; }
      .topbar .close { position: absolute; right: 16px; top: 10px; width: 28px; height: 28px; border: 1px solid rgba(232,235,255,0.3); background: rgba(17,21,46,0.6); color: #e8ebff; border-radius: 6px; }
      .stage-wrap { flex: 1; position: relative; }
      .controls { height: 56px; display: flex; align-items: center; gap: 12px; padding: 0 16px; }
      .controls .icon { width: 28px; height: 28px; border-radius: 6px; border: 1px solid rgba(232,235,255,0.3); background: rgba(17,21,46,0.6); color: #e8ebff; }
      .controls input[type=range] { flex: 1; }
    </style>
    <script src="https://unpkg.com/konva@9/konva.min.js"></script>
  </head>
  <body>
    <div class="wrap">
      <div class="topbar">
        <div id="movingText">Moving index: -</div>
        <button class="close" title="Close">x</button>
      </div>
      <div class="stage-wrap" id="stageWrap"></div>
      <div class="controls">
        <button class="icon" id="playBtn">▶</button>
        <input type="range" id="slider" min="0" max="1" step="0.001" value="0" />
      </div>
    </div>
    <script>
      const data = ${JSON.stringify(payload)};
      const wrap = document.getElementById('stageWrap');
      const movingText = document.getElementById('movingText');
      const playBtn = document.getElementById('playBtn');
      const slider = document.getElementById('slider');
      const gap = 20;
      let simProgress = 0;
      let simPlaying = false;
      const move = 1;
      const hold = 0.2;
      const per = move + hold;
      const zoneIds = Object.keys(data.zone_boundaries || {});
      const getLabel = (zid) => {
        const lbl = data.zone_label_map?.[zid] ?? data.zone_label_map?.[parseInt(zid, 10)] ?? zid;
        const num = Number(lbl);
        return Number.isFinite(num) ? num : Number(zid) || 0;
      };
      zoneIds.sort((a,b) => getLabel(a) - getLabel(b));
      const total = zoneIds.length ? zoneIds.length * per : 1;
      const zoneIndex = {};
      zoneIds.forEach((zid, idx) => zoneIndex[String(zid)] = idx);
      const simLocalFor = (idx) => {
        if (idx == null || idx < 0) return 0;
        const t = simProgress * total - idx * per;
        if (t <= 0) return 0;
        if (t >= per) return 1;
        if (t >= move) return 1;
        const x = t / move;
        return 1 - Math.pow(1 - x, 3);
      };
      const rotatePt = (pt, angleDeg, cx, cy) => {
        if (!angleDeg) return pt;
        const ang = angleDeg * Math.PI / 180;
        const c = Math.cos(ang);
        const s = Math.sin(ang);
        const x = pt[0] - cx;
        const y = pt[1] - cy;
        return [cx + x * c - y * s, cy + x * s + y * c];
      };
      const transformPath = (pts, shift, rot, center) => {
        if (!pts || !pts.length) return [];
        const dx = shift?.[0] ?? 0;
        const dy = shift?.[1] ?? 0;
        const ang = rot ?? 0;
        const cx = center?.[0] ?? 0;
        const cy = center?.[1] ?? 0;
        return pts.map((p) => {
          const r = rotatePt(p, ang, cx, cy);
          return [r[0] + dx, r[1] + dy];
        });
      };
      const toPoints = (pts) => pts.flatMap(p => [p[0], p[1]]);
      const offsetPoints = (pts, dx, dy) => pts.map(p => [p[0] + dx, p[1] + dy]);
      const measureText = (text, size, family) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        ctx.font = size + 'px ' + (family || 'Arial');
        const metrics = ctx.measureText(text || '');
        return { width: metrics.width, height: size };
      };
      const stage = new Konva.Stage({ container: 'stageWrap', width: 10, height: 10 });
      const layerFrame = new Konva.Layer();
      const layerPacked = new Konva.Layer();
      const layerZoneLabels = new Konva.Layer();
      const layerZoneStroke = new Konva.Layer();
      const layerMove = new Konva.Layer();
      stage.add(layerFrame, layerPacked, layerZoneLabels, layerZoneStroke, layerMove);
      const rectLeft = new Konva.Rect({ x: 0, y: 0, width: data.canvas.w, height: data.canvas.h, stroke: '#ffffff', strokeWidth: 1 });
      const rectRight = new Konva.Rect({ x: data.canvas.w + gap, y: 0, width: data.canvas.w, height: data.canvas.h, stroke: '#ffffff', strokeWidth: 1 });
      layerFrame.add(rectLeft, rectRight);

      const packedShapes = [];
      data.regions.forEach((poly, idx) => {
        const zid = data.zone_id?.[idx];
        const zidKey = String(zid);
        const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
        if (!shift) return;
        const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
        const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0,0];
        const packed = transformPath(poly, shift, rot, center);
        const shape = new Konva.Line({
          points: toPoints(packed),
          closed: true,
          fill: data.region_colors[idx] || '#ffffff',
          strokeScaleEnabled: false,
        });
        shape._zidKey = zidKey;
        layerPacked.add(shape);
        packedShapes.push(shape);
      });
      const packedLabels = [];
      (data.packed_labels || []).forEach((lbl) => {
        const size = Math.max((data.font_size || 12) * 0.5, 6);
        const metrics = measureText(lbl.label, size, data.font_family);
        const text = new Konva.Text({
          x: lbl.x,
          y: lbl.y,
          text: lbl.label,
          fill: '#ffffff',
          fontSize: size,
          fontFamily: data.font_family,
          align: 'center',
          verticalAlign: 'middle',
          offsetX: metrics.width / 2,
          offsetY: metrics.height / 2,
        });
        text._zidKey = String(lbl.zid);
        layerPacked.add(text);
        packedLabels.push(text);
      });

      Object.values(data.zone_labels || {}).forEach((lbl) => {
        const size = Math.max((data.font_size || 12) * 0.5, 6);
        const metrics = measureText(lbl.label, size, data.font_family);
        const text = new Konva.Text({
          x: lbl.x + data.canvas.w + gap,
          y: lbl.y,
          text: lbl.label,
          fill: '#ffffff',
          fontSize: size,
          fontFamily: data.font_family,
          align: 'center',
          verticalAlign: 'middle',
          offsetX: metrics.width / 2,
          offsetY: metrics.height / 2,
        });
        layerZoneLabels.add(text);
      });

      zoneIds.forEach((zid) => {
        const paths = data.zone_boundaries?.[zid] || [];
        paths.forEach((p) => {
          const shape = new Konva.Line({
            points: toPoints(offsetPoints(p, data.canvas.w + gap, 0)),
            stroke: '#f5f6ff',
            strokeWidth: 1,
            closed: true,
          });
          layerZoneStroke.add(shape);
        });
      });

      const movingShapes = [];
      data.regions.forEach((poly, idx) => {
        const zid = data.zone_id?.[idx];
        const zidKey = String(zid);
        const shift = data.zone_shift?.[zid] || data.zone_shift?.[parseInt(zid, 10)];
        if (!shift) return;
        const rot = data.zone_rot?.[zid] ?? data.zone_rot?.[parseInt(zid, 10)] ?? 0;
        const center = data.zone_center?.[zid] || data.zone_center?.[parseInt(zid, 10)] || [0,0];
        const src = transformPath(poly, shift, rot, center);
        const dst = offsetPoints(poly, data.canvas.w + gap, 0);
        const shape = new Konva.Line({
          points: toPoints(src),
          closed: true,
          fill: data.region_colors[idx] || '#ffffff',
          strokeScaleEnabled: false,
        });
        shape._zidKey = zidKey;
        shape._src = src;
        shape._dst = dst;
        layerMove.add(shape);
        movingShapes.push(shape);
      });

      const resize = () => {
        const rect = wrap.getBoundingClientRect();
        const totalW = (data.canvas.w * 2) + gap;
        const totalH = data.canvas.h;
        const fitScale = Math.min(rect.width / totalW, rect.height / totalH) * 1.06;
        const offsetX = (rect.width - totalW * fitScale) / 2;
        const offsetY = (rect.height - totalH * fitScale) / 2;
        stage.width(rect.width);
        stage.height(rect.height);
        stage.scale({ x: fitScale, y: fitScale });
        stage.position({ x: offsetX, y: offsetY });
        stage.draw();
      };
      resize();
      window.addEventListener('resize', resize);

      const update = () => {
        const activeIdx = zoneIds.length ? Math.min(zoneIds.length - 1, Math.max(0, Math.floor((simProgress * total) / per))) : -1;
        const activeZid = activeIdx >= 0 ? zoneIds[activeIdx] : null;
        const activeLabel = activeZid != null ? (data.zone_label_map?.[activeZid] ?? data.zone_label_map?.[parseInt(activeZid, 10)] ?? activeZid) : '-';
        movingText.textContent = 'Moving index: ' + activeLabel;
        packedShapes.forEach((shape) => {
          const idx = zoneIndex[shape._zidKey] ?? 0;
          const local = simLocalFor(idx);
          shape.visible(local <= 0);
        });
        packedLabels.forEach((shape) => {
          const idx = zoneIndex[shape._zidKey] ?? 0;
          const local = simLocalFor(idx);
          shape.visible(local <= 0);
        });
        movingShapes.forEach((shape) => {
          const idx = zoneIndex[shape._zidKey] ?? 0;
          const local = simLocalFor(idx);
          if (local <= 0) {
            shape.visible(false);
            return;
          }
          const pts = shape._src.map((sp, k) => {
            const dp = shape._dst[k] || sp;
            return [sp[0] + (dp[0] - sp[0]) * local, sp[1] + (dp[1] - sp[1]) * local];
          });
          shape.points(toPoints(pts));
          shape.visible(true);
        });
        layerPacked.draw();
        layerMove.draw();
      };

      const tick = (now) => {
        if (!simPlaying) return;
        const dt = 1 / 60;
        simProgress = Math.min(1, simProgress + dt / total);
        slider.value = simProgress.toFixed(3);
        if (simProgress >= 1) simPlaying = false;
        update();
        requestAnimationFrame(tick);
      };
      playBtn.addEventListener('click', () => {
        simPlaying = !simPlaying;
        playBtn.textContent = simPlaying ? '❚❚' : '▶';
        if (simPlaying) requestAnimationFrame(tick);
      });
      slider.addEventListener('input', (e) => {
        simProgress = parseFloat(e.target.value || '0');
        update();
      });
      update();
    </script>
  </body>
</html>`;
  };

  const handleWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = stageRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setScale(newScale);
    setPos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleRegionWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = regionRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setRegionScale(newScale);
    setRegionPos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleMainViewWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = e.target.getStage ? e.target.getStage() : null;
    if (!stage) return;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setMainViewScale(newScale);
    setMainViewPos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleMainViewDragMove = (e) => {
    const stage = e.target.getStage ? e.target.getStage() : e.target;
    if (!stage) return;
    setMainViewPos({ x: stage.x(), y: stage.y() });
  };

  const handleRegionDragMove = (e) => {
    const stage = e.target.getStage ? e.target.getStage() : e.target;
    if (!stage) return;
    setRegionPos({ x: stage.x(), y: stage.y() });
  };

  const segmentWouldIntersect = (aIdx, bIdx) => {
    if (!nodes[aIdx] || !nodes[bIdx]) return true;
    const a1 = [nodes[aIdx].x, nodes[aIdx].y];
    const a2 = [nodes[bIdx].x, nodes[bIdx].y];
    for (const [s0, s1] of segs) {
      if (s0 === aIdx || s1 === aIdx || s0 === bIdx || s1 === bIdx) {
        continue;
      }
      const b1 = [nodes[s0].x, nodes[s0].y];
      const b2 = [nodes[s1].x, nodes[s1].y];
      const inter = segmentIntersect(a1, a2, b1, b2);
      if (!inter) continue;
      if (inter.t > 1e-6 && inter.t < 1 - 1e-6 && inter.u > 1e-6 && inter.u < 1 - 1e-6) {
        return true;
      }
    }
    return false;
  };

  const findEdgeCandidate = (worldPt) => {
    if (!nodes.length) return null;
    const EDGE_HOVER_DIST = 10;
    const EDGE_MAX_LEN = 80;
    const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dx = nodes[j].x - nodes[i].x;
        const dy = nodes[j].y - nodes[i].y;
        const len = Math.hypot(dx, dy);
        if (len > EDGE_MAX_LEN) continue;
        if (segSet.has(edgeKey(i, j))) continue;
        const mx = (nodes[i].x + nodes[j].x) * 0.5;
        const my = (nodes[i].y + nodes[j].y) * 0.5;
        const d = Math.hypot(worldPt.x - mx, worldPt.y - my);
        if (d > EDGE_HOVER_DIST || d >= bestDist) continue;
        if (segmentWouldIntersect(i, j)) continue;
        bestDist = d;
        best = { a: i, b: j };
      }
    }
    return best;
  };

  const findExistingEdgeCandidate = (worldPt) => {
    if (!nodes.length || !segs.length) return null;
    const EDGE_HOVER_DIST = 10;
    let best = null;
    let bestDist = Infinity;
    segs.forEach(([a, b], idx) => {
      const na = nodes[a];
      const nb = nodes[b];
      if (!na || !nb) return;
      const d = pointSegDist([worldPt.x, worldPt.y], [na.x, na.y], [nb.x, nb.y]);
      if (d <= EDGE_HOVER_DIST && d < bestDist) {
        bestDist = d;
        best = { a, b, idx };
      }
    });
    return best;
  };

  const pruneIsolatedNodes = (nextNodes, nextSegs) => {
    const connected = new Set();
    nextSegs.forEach(([a, b]) => {
      connected.add(a);
      connected.add(b);
    });
    const remap = new Map();
    const kept = [];
    nextNodes.forEach((n, idx) => {
      if (connected.has(idx)) {
        remap.set(idx, kept.length);
        kept.push({ ...n, id: kept.length });
      }
    });
    const remappedSegs = nextSegs
      .map(([a, b]) => [remap.get(a), remap.get(b)])
      .filter(([a, b]) => a != null && b != null && a !== b);
    return { nodes: kept, segs: remappedSegs };
  };

  const findZoneAtPoint = (pt) => {
    const source = zoneScene || scene;
    const zones = source?.zone_boundaries || {};
    for (const [zid, paths] of Object.entries(zones)) {
      for (const poly of paths || []) {
        if (pointInPoly([pt.x, pt.y], poly)) return String(zid);
      }
    }
    return null;
  };

  const findZoneIdByLabel = (label) => {
    const source = zoneScene || scene;
    if (!source) return String(label);
    const target = String(label);
    const map = source.zone_label_map || {};
    for (const [zid, mapped] of Object.entries(map)) {
      if (String(mapped) === target) return String(zid);
    }
    for (const [zid, info] of Object.entries(source.zone_labels || {})) {
      if (String(info?.label) === target) return String(zid);
    }
    return target;
  };

  const getZoneAlias = (zid, source) => {
    if (!source) return String(zid);
    const mapped =
      source.zone_label_map?.[zid] ??
      source.zone_label_map?.[parseInt(zid, 10)];
    if (mapped != null) return String(mapped);
    const label =
      source.zone_labels?.[zid]?.label ??
      source.zone_labels?.[parseInt(zid, 10)]?.label;
    if (label != null) return String(label);
    return String(zid);
  };

  const packedBoxData = useMemo(() => {
    if (!scene?.zone_pack_polys || !scene?.zone_order || !scene?.placements) return [];
    const out = [];
    const zoneOrder = scene.zone_order || [];
    zoneOrder.forEach((zid, idx) => {
      const poly = scene.zone_pack_polys?.[idx];
      const placement = scene.placements?.[idx];
      if (!poly || !poly.length || !placement) return;
      const dx = placement?.[0] ?? -1;
      const dy = placement?.[1] ?? -1;
      const bw = placement?.[2] ?? 0;
      const bh = placement?.[3] ?? 0;
      if (bw <= 0 || bh <= 0) return;
      const info = scene.rot_info?.[idx] || {};
      const ang = info?.angle ?? 0;
      const cx = info?.cx ?? 0;
      const cy = info?.cy ?? 0;
      const rpts = poly.map((p) => rotatePt(p, ang, cx, cy));
      const bb = bboxFromPts(rpts);
      if (!bb) return;
      const x = dx + bb.minx;
      const y = dy + bb.miny;
      const bin = scene?.placement_bin?.[zid] ?? scene?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (scene?.canvas?.w || 0) + 40 : 0;
      out.push({
        zid: String(zid),
        label: getZoneAlias(zid, scene),
        minx: x,
        miny: y,
        maxx: x + bw,
        maxy: y + bh,
        xOffset,
      });
    });
    return out;
  }, [scene]);

  const packedSource = zoneScene || scene;
  const hasManualPackedEdits = useMemo(
    () => Object.keys(manualPackedEdits || {}).length > 0,
    [manualPackedEdits]
  );

  const packedLiveFillItems = useMemo(() => {
    if (!hasManualPackedEdits || !packedSource?.regions || !packedSource?.zone_id) return [];
    const packedColors = packedSource.pack_region_colors || packedSource.region_colors || [];
    const out = [];
    packedSource.regions.forEach((poly, idx) => {
      if (!poly || poly.length < 3) return;
      const zidRaw = packedSource.zone_id?.[idx];
      if (zidRaw == null) return;
      const zid = String(zidRaw);
      const shift =
        packedSource.zone_shift?.[zid] ||
        packedSource.zone_shift?.[parseInt(zid, 10)] ||
        [0, 0];
      const rot =
        packedSource.zone_rot?.[zid] ??
        packedSource.zone_rot?.[parseInt(zid, 10)] ??
        0;
      const center =
        packedSource.zone_center?.[zid] ||
        packedSource.zone_center?.[parseInt(zid, 10)] ||
        [0, 0];
      const moved = transformPath(poly, shift, rot, center);
      if (!moved || moved.length < 3) return;
      const bin =
        packedSource?.placement_bin?.[zid] ??
        packedSource?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;
      const fill = packedColors[idx] || "#808080";
      out.push({
        key: `pflive-${idx}`,
        points: toPoints(offsetPoints(moved, xOffset, 0)),
        fill,
      });
    });
    return out;
  }, [hasManualPackedEdits, packedSource]);

  const packedZoneColorByZid = useMemo(() => {
    const source = packedSource || scene;
    const packedColors = source?.pack_region_colors || source?.region_colors || [];
    const out = {};
    if (!source?.zone_id || !packedColors.length) return out;
    for (let i = 0; i < source.zone_id.length; i++) {
      const zid = String(source.zone_id[i]);
      if (out[zid]) continue;
      out[zid] = packedColors[i] || "#808080";
    }
    return out;
  }, [packedSource, scene]);

  const packedLiveBleedItems = useMemo(() => {
    if (!enableBleed) return [];
    if (!hasManualPackedEdits || !packedSource?.zone_pack_polys || !packedSource?.zone_order) return [];
    const out = [];
    const zoneOrder = packedSource.zone_order || [];
    zoneOrder.forEach((zidRaw, idx) => {
      const zid = String(zidRaw);
      const poly = packedSource.zone_pack_polys?.[idx];
      if (!poly || poly.length < 3) return;
      const shift =
        packedSource.zone_shift?.[zid] ||
        packedSource.zone_shift?.[parseInt(zid, 10)] ||
        [0, 0];
      const rot =
        packedSource.zone_rot?.[zid] ??
        packedSource.zone_rot?.[parseInt(zid, 10)] ??
        0;
      const center =
        packedSource.zone_center?.[zid] ||
        packedSource.zone_center?.[parseInt(zid, 10)] ||
        [0, 0];
      const moved = transformPath(poly, shift, rot, center);
      if (!moved || moved.length < 3) return;
      const bin =
        packedSource?.placement_bin?.[zid] ??
        packedSource?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;
      out.push({
        key: `pblive-${zid}-${idx}`,
        points: toPoints(offsetPoints(moved, xOffset, 0)),
        fill: packedZoneColorByZid[zid] || "#808080",
        stroke: "rgba(255,255,255,0.95)",
      });
    });
    return out;
  }, [enableBleed, hasManualPackedEdits, packedSource, packedZoneColorByZid]);

  const packedEmptyCellsDerived = useMemo(() => {
    if (!packedSource) return [];
    const packedPolyData = buildPackedPolyData(packedSource);
    return buildPackedEmptyCells(packedSource, packedPolyData);
  }, [packedSource]);

  const packedZoneOutlineItems = useMemo(() => {
    if (!packedSource?.zone_polys || !packedSource?.zone_order) return [];
    const out = [];
    (packedSource.zone_order || []).forEach((zidRaw, idx) => {
      const zid = String(zidRaw);
      const poly = packedSource.zone_polys?.[idx];
      if (!poly || poly.length < 3) return;
      const shift =
        packedSource.zone_shift?.[zid] ||
        packedSource.zone_shift?.[parseInt(zid, 10)];
      if (!shift) return;
      const rot =
        packedSource.zone_rot?.[zid] ??
        packedSource.zone_rot?.[parseInt(zid, 10)] ??
        0;
      const center =
        packedSource.zone_center?.[zid] ||
        packedSource.zone_center?.[parseInt(zid, 10)] ||
        [0, 0];
      const tpts = transformPath(poly, shift, rot, center);
      if (!tpts || tpts.length < 3) return;
      const bin =
        packedSource?.placement_bin?.[zid] ??
        packedSource?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;
      out.push({
        zid,
        idx,
        page,
        xOffset,
        pts: tpts,
        offsetPts: offsetPoints(tpts, xOffset, 0),
      });
    });
    return out;
  }, [packedSource]);

  

  const packedCellsByBin = useMemo(() => {
    if (!packedEmptyCellsDerived.length || !packedSource?.canvas) {
      return { 0: packedEmptyCellsDerived, 1: [] };
    }
    const offset = (packedSource.canvas.w || 0) + 40;
    const cells1 = packedEmptyCellsDerived.map((c) => [c[0] + offset, c[1]]);
    return { 0: packedEmptyCellsDerived, 1: cells1 };
  }, [packedEmptyCellsDerived, packedSource]);

  const packedLabelSnappedAll = useMemo(() => {
    // Keep render deterministic: no extra snapping/adjustment after raster->vector conversion.
    return [];
  }, [packedSource]);

  const packedIndexItems = useMemo(() => {
    if (!packedSource) return [];
    const out = [];
    const occupied = [];
    const intersects = (a, b) =>
      !(a.x + a.w < b.x || b.x + b.w < a.x || a.y + a.h < b.y || b.y + b.h < a.y);
    const pageW = packedSource?.canvas?.w || 0;
    const pageH = packedSource?.canvas?.h || 0;
    Object.entries(packedSource.zone_labels || {}).forEach(([zid, lbl]) => {
      if (!lbl) return;
      const shift =
        packedSource.zone_shift?.[zid] ?? packedSource.zone_shift?.[parseInt(zid, 10)];
      if (!shift) return;
      const rot =
        packedSource.zone_rot?.[zid] ??
        packedSource.zone_rot?.[parseInt(zid, 10)] ??
        0;
      const center =
        packedSource.zone_center?.[zid] ??
        packedSource.zone_center?.[parseInt(zid, 10)] ??
        [0, 0];
      const zonePoly = packedZoneOutlineItems.find((item) => String(item.zid) === String(zid));
      if (!zonePoly?.pts?.length) return;
      const bin =
        packedSource?.placement_bin?.[zid] ??
        packedSource?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;
      const label = `${getZoneAlias(zid, packedSource)}`;
      const size = Math.max(labelFontSize / Math.max(regionScale, 0.0001), 6 / Math.max(regionScale, 0.0001));
      const metrics = measureText(label, size, labelFontFamily);
      const halfW = metrics.width / 2;
      const halfH = metrics.height / 2;

      const bestPath = zonePoly.pts;
      if (!bestPath || bestPath.length < 2) return;

      const { area: polyArea } = polyAreaAndCentroid(bestPath);
      const candidates = [];
      for (let i = 0; i < bestPath.length; i++) {
        const p0 = bestPath[i];
        const p1 = bestPath[(i + 1) % bestPath.length];
        const vx = p1[0] - p0[0];
        const vy = p1[1] - p0[1];
        const len = Math.hypot(vx, vy);
        if (len < 1e-6) continue;
        const nx = polyArea >= 0 ? vy / len : -vy / len;
        const ny = polyArea >= 0 ? -vx / len : vx / len;
        const tx = vx / len;
        const ty = vy / len;
        const mx = 0.5 * (p0[0] + p1[0]);
        const my = 0.5 * (p0[1] + p1[1]);
        for (const d of [7, 11, 15]) {
          for (const s of [0, 6, -6, 12, -12]) {
            candidates.push({ x: mx + tx * s + nx * d + xOffset, y: my + ty * s + ny * d, edgeLen: len });
          }
        }
      }
      candidates.sort((a, b) => b.edgeLen - a.edgeLen);

      let chosen = null;
      for (const c of candidates) {
        if (c.x - halfW < 0 || c.x + halfW > pageW + (page === 1 ? pageW + 40 : 0)) continue;
        if (c.y - halfH < 0 || c.y + halfH > pageH) continue;
        const box = { x: c.x - halfW, y: c.y - halfH, w: metrics.width, h: metrics.height };
        if (occupied.some((b) => intersects(box, b))) continue;
        chosen = c;
        occupied.push(box);
        break;
      }
      if (!chosen) {
        const [pt] = transformPath([[lbl.x, lbl.y]], shift, rot, center);
        if (!pt) return;
        chosen = { x: pt[0] + xOffset, y: pt[1] };
      }
      out.push({
        zid: String(zid),
        label,
        x: chosen.x,
        y: chosen.y,
      });
    });
    return out;
  }, [packedSource, packedZoneOutlineItems, packedLabelSnappedAll, labelFontFamily, labelFontSize, regionScale]);

  const packedLowAreaWarnings = useMemo(() => {
    // Disable warning markers while validating raster->vector flow.
    return [];
    if (!packedSource?.zone_boundaries) return [];
    const zoneCandidates = [];
    Object.entries(packedSource.zone_boundaries || {}).forEach(([zid, paths]) => {
      if (!paths || !paths.length) return;
      let zoneArea = 0;
      (paths || []).forEach((p) => {
        const { area } = polyAreaAndCentroid(p);
        zoneArea += Math.abs(area || 0);
      });
      if (!(zoneArea > 0)) return;
      zoneCandidates.push({ zid, paths, zoneArea });
    });
    if (!zoneCandidates.length) return [];

    const smallest = zoneCandidates
      .sort((a, b) => a.zoneArea - b.zoneArea)
      .slice(0, 10);

    const avgArea = zoneAreaStats.avg || 0;
    const out = [];
    smallest.forEach(({ zid, paths, zoneArea }) => {
      const shift =
        packedSource.zone_shift?.[zid] || packedSource.zone_shift?.[parseInt(zid, 10)];
      if (!shift) return;
      const rot =
        packedSource.zone_rot?.[zid] ??
        packedSource.zone_rot?.[parseInt(zid, 10)] ??
        0;
      const center =
        packedSource.zone_center?.[zid] ||
        packedSource.zone_center?.[parseInt(zid, 10)] ||
        [0, 0];
      const bin =
        packedSource?.placement_bin?.[zid] ??
        packedSource?.placement_bin?.[parseInt(zid, 10)];
      const page = bin === 1 ? 1 : 0;
      const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;

      let minx = Infinity;
      let miny = Infinity;
      let maxx = -Infinity;
      let maxy = -Infinity;
      let found = false;
      (paths || []).forEach((p) => {
        const tpts = transformPath(p, shift, rot, center);
        (tpts || []).forEach((pt) => {
          found = true;
          const x = pt[0] + xOffset;
          const y = pt[1];
          if (x < minx) minx = x;
          if (x > maxx) maxx = x;
          if (y < miny) miny = y;
          if (y > maxy) maxy = y;
        });
      });
      if (!found) return;

      out.push({
        zid: String(zid),
        label: getZoneAlias(zid, packedSource),
        ratio: avgArea > 0 ? zoneArea / avgArea : 0,
        page,
        minx,
        miny,
        maxx,
        maxy,
      });
    });
    return out;
  }, [packedSource, zoneAreaStats.avg]);


  const findRegionAtPoint = (regions, pt) => {
    if (!regions || !pt) return -1;
    for (let i = 0; i < regions.length; i++) {
      if (pointInPoly([pt.x, pt.y], regions[i])) return i;
    }
    return -1;
  };

  const normalizeClickCache = (payload) => {
    const raw = Array.isArray(payload) ? payload : payload && Array.isArray(payload.clicks) ? payload.clicks : [];
    return raw.filter(
      (item) =>
        item &&
        Number.isFinite(item.rid) &&
        Number.isFinite(item.attach_to)
    ).map((item) => ({
      rid: Math.trunc(item.rid),
      attach_to: Math.trunc(item.attach_to),
      ...(Number.isFinite(item.x) && Number.isFinite(item.y)
        ? { x: item.x, y: item.y }
        : {}),
    }));
  };

  const computePackedSignature = (source) => {
    if (!source) return "none";
    let h = 2166136261;
    const mix = (n) => {
      let x = Number.isFinite(n) ? Math.floor(n * 1000) : 0;
      h ^= x >>> 0;
      h = Math.imul(h, 16777619) >>> 0;
    };
    mix(source?.canvas?.w || 0);
    mix(source?.canvas?.h || 0);
    const regions = source?.regions || [];
    const zid = source?.zone_id || [];
    mix(regions.length);
    mix(zid.length);
    for (let i = 0; i < Math.min(64, zid.length); i++) mix(zid[i]);
    for (let i = 0; i < Math.min(24, regions.length); i++) {
      const r = regions[i] || [];
      mix(r.length);
      for (let j = 0; j < Math.min(3, r.length); j++) {
        mix(r[j]?.[0]);
        mix(r[j]?.[1]);
      }
    }
    return `p${h.toString(16)}`;
  };

  const applyManualPackedEdits = (source, edits) => {
    if (!source || !edits || !Object.keys(edits).length) return source;
    const next = { ...source };
    next.zone_shift = { ...(source.zone_shift || {}) };
    next.zone_rot = { ...(source.zone_rot || {}) };
    Object.entries(edits).forEach(([zid, v]) => {
      if (!v) return;
      if (Number.isFinite(v.dx) && Number.isFinite(v.dy)) {
        next.zone_shift[zid] = [v.dx, v.dy];
      }
      if (Number.isFinite(v.rot)) {
        next.zone_rot[zid] = v.rot;
      }
    });
    return next;
  };

  const patchStateJson = async (patch, sourceName = selectedSource) => {
    try {
      let state = {};
      try {
        const r = await fetch(`/api/state?${sourceQuery(sourceName)}`);
        if (r.ok) state = (await r.json()) || {};
      } catch {}
      const next = { ...(state || {}), ...(patch || {}) };
      await fetch(`/api/state?${sourceQuery(sourceName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(next),
      });
    } catch {}
  };

  const persistSourceCacheNow = async (
    nextNodes = nodes,
    nextSegs = segs,
    nextVoronoi = sourceVoronoi,
    sourceName = selectedSource
  ) => {
    await patchStateJson(
      {
        svg_nodes: normalizeNodesForSave(nextNodes, sourceScale || 1),
        svg_segments: (nextSegs || []).map(([a, b]) => [a, b]),
        source_voronoi: normalizeVoronoiForSave(nextVoronoi, sourceScale || 1),
      },
      sourceName
    );
  };

  const saveSourceEditsAndRefreshRegion = async (sourceName = selectedSource) => {
    try {
      setError("");
      setSceneLoading(true);
      await persistSourceCacheNow(nodes, segs, sourceVoronoi, sourceName);
      const res = await fetch(
        `/api/source_region_scene?${sourceQuery(sourceName)}&count=120&relax=2&seed=7&_ts=${Date.now()}`,
        { cache: "no-store" }
      );
      if (!res.ok) {
        throw new Error(`source region scene fetch failed: ${res.status}`);
      }
      const regionData = await res.json();
      const scaledRegionData =
        (sourceScale || 1) === 1 ? regionData : scaleSceneData(regionData, sourceScale || 1);
      setScene(scaledRegionData);
      const nextZoneScene = buildSnapZoneSceneFromRegionScene(scaledRegionData);
      if (nextZoneScene?.zone_id?.length) {
        setZoneScene(nextZoneScene);
        setLeftTab("region");
        clearManualPackedEdits(sourceName);
        await refreshPackedFromZoneScene(nextZoneScene, enableBleed, sourceName);
      }
      setExportMsg("Saved Region + Zone");
      setTimeout(() => setExportMsg(""), 2000);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSceneLoading(false);
    }
  };

  const resetSourceCacheAndReload = async (sourceName = selectedSource) => {
    try {
      setError("");
      setSceneLoading(true);
      const res = await fetch(`/api/reset_source_cache?${sourceQuery(sourceName)}`, {
        method: "POST",
      });
      if (!res.ok) {
        throw new Error(`reset source cache failed: ${res.status}`);
      }
      resetSourceHistory();
      await loadScene(true, false, false, false, sourceName);
      setExportMsg("Reset Source");
      setTimeout(() => setExportMsg(""), 2000);
    } catch (err) {
      setError(err.message || String(err));
      setSceneLoading(false);
    }
  };

  const scheduleSaveManualPackedEdits = (edits, source, sourceName = selectedSource) => {
    if (manualPackedSaveTimerRef.current) {
      clearTimeout(manualPackedSaveTimerRef.current);
    }
    manualPackedSaveTimerRef.current = setTimeout(() => {
      const payload = {
        manual_packed: {
          signature: computePackedSignature(source),
          edits,
        },
      };
      patchStateJson(payload, sourceName);
    }, 180);
  };

  const clearManualPackedEdits = (sourceName = selectedSource) => {
    setManualPackedEdits({});
    patchStateJson({ manual_packed: null }, sourceName);
  };

  const refreshPackedFromZoneScene = async (
    source,
    bleedEnabled = enableBleed,
    sourceName = selectedSource
  ) => {
    if (!source?.regions || !source?.zone_id || !source?.canvas) return;
    const startTs = performance.now();
    setPackTiming((prev) => ({
      ...prev,
      running: true,
      startTs,
      elapsedMs: 0,
    }));
    try {
      const res = await fetch(`/api/pack_from_scene?${sourceQuery(sourceName)}&_ts=${Date.now()}`, {
        method: "POST",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: sourceName,
          canvas: source.canvas,
          regions: source.regions,
          zone_id: source.zone_id,
          zone_polys: source.zone_polys || [],
          zone_order: source.zone_order || [],
          region_colors: source.pack_region_colors || source.region_colors || [],
          pack_padding: packPadding,
          pack_margin_x: packMarginX,
          pack_margin_y: packMarginY,
          pack_bleed: packBleed,
          draw_scale: drawScale,
          pack_grid: packGrid,
          pack_angle: packAngle,
          pack_mode: packMode,
        }),
      });
      if (!res.ok) {
        throw new Error(`pack_from_scene failed: ${res.status}`);
      }
      const data = await res.json();
      setShowRasterTemp(false);
      if (data?.packed_svg) {
        const parsed = parsePackedSvg(data.packed_svg);
        setPackedFillPaths(parsed.fillPaths);
        setPackedBleedPaths(parsed.bleedPaths);
        setPackedBleedError(parsed.hasBleed ? "" : "packed.svg missing bleed layer");
      }
      if (data?.packed_svg_page2) {
        const parsed2 = parsePackedSvg(data.packed_svg_page2);
        setPackedFillPaths2(parsed2.fillPaths);
        setPackedBleedPaths2(parsed2.bleedPaths);
        setPackedBleedError2(parsed2.hasBleed ? "" : "packed_page2.svg missing bleed layer");
      } else {
        setPackedFillPaths2([]);
        setPackedBleedPaths2([]);
        setPackedBleedError2("");
      }
      if (data?.zone_shift || data?.zone_rot || data?.zone_center || data?.placement_bin) {
        setZoneScene((prev) =>
          prev
            ? {
                ...prev,
                zone_polys: data.zone_polys || prev.zone_polys,
                zone_order: data.zone_order || prev.zone_order,
                zone_pack_polys: data.zone_pack_polys || prev.zone_pack_polys,
                zone_shift: data.zone_shift || prev.zone_shift,
                zone_rot: data.zone_rot || prev.zone_rot,
                zone_center: data.zone_center || prev.zone_center,
                placement_bin: data.placement_bin || prev.placement_bin,
              }
            : prev
        );
      }
      const tTotal = data?.timings_ms?.total;
      if (data?.one_page_ok === true) {
        setPackUiLog(
          `Pack: One page OK${Number.isFinite(tTotal) ? ` (${Math.round(tTotal)} ms)` : ""}`
        );
      } else if (Number.isFinite(data?.unplaced_count) && data.unplaced_count > 0) {
        setPackUiLog(
          `Pack: WARNING overflow, unplaced=${data.unplaced_count}${
            Number.isFinite(tTotal) ? ` (${Math.round(tTotal)} ms)` : ""
          }`
        );
      } else if (Number.isFinite(tTotal)) {
        setPackUiLog(`Pack: ${Math.round(tTotal)} ms`);
      }
      return data;
    } catch (err) {
      setPackedBleedError(err.message || String(err));
      setPackUiLog(`Pack: failed (${err.message || String(err)})`);
      return null;
    } finally {
      const durationMs = Math.max(0, performance.now() - startTs);
      setPackTiming((prev) => {
        const nextCount = prev.count + 1;
        const nextAvg =
          prev.avgMs == null ? durationMs : (prev.avgMs * prev.count + durationMs) / nextCount;
        return {
          ...prev,
          running: false,
          startTs: 0,
          elapsedMs: durationMs,
          lastMs: durationMs,
          avgMs: nextAvg,
          count: nextCount,
        };
      });
    }
  };

  const getPackedSourceForCompute = async (sourceName = selectedSource) => {
    // Always use the latest source_region_scene (option 1), then rebuild snap-zone scene from it.
    const sq = sourceQuery(sourceName);
    const regionRes = await fetch(
      `/api/source_region_scene?${sq}&count=120&relax=2&seed=7&_ts=${Date.now()}`,
      { cache: "no-store" }
    );
    if (!regionRes.ok) {
      throw new Error(`source region scene fetch failed: ${regionRes.status}`);
    }
    const regionData = await regionRes.json();
    const scaleRatio = sourceScale || 1;
    const scaledRegion = scaleRatio === 1 ? regionData : scaleSceneData(regionData, scaleRatio);
    setScene(scaledRegion);
    const rebuilt = buildSnapZoneSceneFromRegionScene(scaledRegion);
    if (!rebuilt?.regions?.length || !rebuilt?.zone_id?.length) {
      throw new Error("snap zone scene missing from latest source_region_scene");
    }
    setZoneScene(rebuilt);
    return rebuilt;
  };

  const repackCurrentPacked = async (bleedEnabled = enableBleed) => {
    try {
      const source = await getPackedSourceForCompute(selectedSource);
      setError("");
      setSceneLoading(true);
      setShowRasterTemp(false);
      clearManualPackedEdits(selectedSource);
      await refreshPackedFromZoneScene(source, bleedEnabled, selectedSource);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSceneLoading(false);
    }
  };

  const computePackedFromCurrentZone = async () => {
    if (computeBusy) return;
    try {
      const source = await getPackedSourceForCompute(selectedSource);
      setError("");
      setComputeBusy(true);
      setShowRasterTemp(false);
      await refreshPackedFromZoneScene(source, enableBleed, selectedSource);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setComputeBusy(false);
    }
  };

  const packFromRegionSnap = async () => {
    if (!scene?.regions?.length || !scene?.snap_region_map) return;
    if (computeBusy) return;
    try {
      setError("");
      setComputeBusy(true);
      setSceneLoading(true);
      setShowRasterTemp(false);
      const source = buildSnapZoneSceneFromRegionScene(scene);
      if (!source?.zone_id?.length) {
        throw new Error("snap zone scene missing");
      }
      setZoneScene(source);
      clearManualPackedEdits(selectedSource);
      await refreshPackedFromZoneScene(source, enableBleed, selectedSource);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSceneLoading(false);
      setComputeBusy(false);
    }
  };

  const showTempRasterFromCurrentZone = async () => {
    if (rasterBusy) return;
    try {
      const source = await getPackedSourceForCompute(selectedSource);
      setError("");
      setRasterBusy(true);
      const res = await fetch(`/api/pack_from_scene?${sourceQuery(selectedSource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: selectedSource,
          canvas: source.canvas,
          regions: source.regions,
          zone_id: source.zone_id,
          region_colors: source.region_colors || [],
          pack_padding: packPadding,
          pack_margin_x: packMarginX,
          pack_margin_y: packMarginY,
          pack_bleed: packBleed,
          draw_scale: drawScale,
          pack_grid: packGrid,
          pack_angle: packAngle,
          pack_mode: packMode,
          raster_only: true,
        }),
      });
      if (!res.ok) throw new Error(`raster compute failed: ${res.status}`);
      const data = await res.json();
      const url = data?.raster_tmp_png_url || "/out/tmp_raster_pack.png";
      setRasterTempSrc(`${url}?t=${Date.now()}`);
      setShowRasterTemp(true);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setRasterBusy(false);
    }
  };

  const getZoneStageWorldPoint = (evt) => {
    const stage = evt?.target?.getStage?.() || zoneRef.current;
    if (!stage) return null;
    const pointer = stage.getPointerPosition?.();
    if (!pointer) return null;
    const scale = stage.scaleX?.() || mainViewScale || 1;
    const x = (pointer.x - stage.x()) / scale;
    const y = (pointer.y - stage.y()) / scale;
    return { x, y };
  };

  const saveZoneClickCache = async (entry) => {
    if (!entry || typeof entry !== "object") return;
    const row = {};
    if (Number.isFinite(entry.x) && Number.isFinite(entry.y)) {
      row.x = entry.x;
      row.y = entry.y;
    }
    if (Number.isFinite(entry.rid)) row.rid = Math.trunc(entry.rid);
    if (Number.isFinite(entry.attach_to)) row.attach_to = Math.trunc(entry.attach_to);
    if (!Number.isFinite(row.rid) || !Number.isFinite(row.attach_to)) return;
    const deterministic = (zoneClickCacheRef.current || []).filter(
      (item) => Number.isFinite(item?.rid) && Number.isFinite(item?.attach_to)
    );
    const next = [...deterministic, row];
    zoneClickCacheRef.current = next;
    try {
      await fetch(`/api/source_zone_click?${sourceQuery(selectedSource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ clicks: next }),
      });
    } catch {
      // ignore cache write errors
    }
  };

  const applyZoneDetachAttach = (source, rid, adj, cursorMap) => {
    if (!source?.regions || !source?.zone_id) return { source, changed: false };
    if (rid == null || rid < 0 || rid >= source.regions.length) return { source, changed: false };
    const currentZid = source.zone_id[rid];
    const neighbors = adj?.[rid] || [];
    if (!neighbors.length) return { source, changed: false };
    const neighborZids = [];
    const seen = new Set();
    for (const nb of neighbors) {
      const zid = source.zone_id[nb];
      if (zid == null) continue;
      if (String(zid) === String(currentZid)) continue;
      const key = String(zid);
      if (seen.has(key)) continue;
      seen.add(key);
      neighborZids.push(zid);
    }
    if (!neighborZids.length) return { source, changed: false };
    neighborZids.sort((a, b) => Number(a) - Number(b));
    const cursor = cursorMap[rid] || 0;
    const targetZid = neighborZids[cursor % neighborZids.length];
    cursorMap[rid] = (cursor + 1) % neighborZids.length;
    const nextZoneId = source.zone_id.slice();
    nextZoneId[rid] = targetZid;
    const nextBoundaries = buildZoneBoundaries(source.regions, nextZoneId, 0);
    return {
      source: { ...source, zone_id: nextZoneId, zone_boundaries: nextBoundaries },
      changed: true,
      targetZid,
      currentZid,
      neighborZids,
    };
  };

  const applyZoneClickCache = (base, clicks) => {
    if (!base?.regions || !base?.zone_id || !clicks?.length) return base;
    const adj = buildRegionAdjacencyMulti(base.regions || [], [neighborSnap, 2]);
    const cursorMap = {};
    let nextSource = base;
    let lastTarget = null;
    clicks.forEach((pt, idx) => {
      if (!pt) return;
      if (Number.isFinite(pt.rid) && Number.isFinite(pt.attach_to)) {
        const rid = Math.trunc(pt.rid);
        const target = Math.trunc(pt.attach_to);
        if (rid >= 0 && rid < (nextSource.zone_id || []).length) {
          const cur = nextSource.zone_id[rid];
          if (String(cur) !== String(target)) {
            const nextZoneId = (nextSource.zone_id || []).slice();
            nextZoneId[rid] = target;
            const nextBoundaries = buildZoneBoundaries(nextSource.regions, nextZoneId, 0);
            nextSource = { ...nextSource, zone_id: nextZoneId, zone_boundaries: nextBoundaries };
            lastTarget = target;
            return;
          }
        }
      }
      if (!Number.isFinite(pt.x) || !Number.isFinite(pt.y)) return;
      const rid = findRegionAtPoint(nextSource.regions, pt);
      if (rid < 0) return;
      const res = applyZoneDetachAttach(nextSource, rid, adj, cursorMap);
      if (!res.changed) return;
      nextSource = res.source;
      lastTarget = res.targetZid;
    });
    if (lastTarget != null) setSelectedZoneId(String(lastTarget));
    return nextSource;
  };

  const renderLeftDebug = () => (
    <div className="left-debug">
      <div className="zone-count">
        Zones: {scene?.zone_id ? Math.max(...scene.zone_id) + 1 : 0}
      </div>
      <div className="zone-count">
        Debug:
        {scene?.debug
          ? ` raw=${scene.debug.polygons_raw || 0} kept=${scene.debug.polygons_final || 0} small=${scene.debug.polygons_removed_small || 0} largest=${scene.debug.polygons_removed_largest || 0} tri_keep=${scene.debug.tri_kept || 0} tri_small=${scene.debug.tri_removed_small || 0} tri_out=${scene.debug.tri_removed_outside || 0} packed=${scene.debug.packed_placed || 0}/${scene.debug.zones_total || 0}`
          : " n/a"}
      </div>
      <div className="zone-count">
        ZonePoly:
        {scene?.debug
          ? ` empty=${(scene.debug.zones_empty || []).length} hull=${(scene.debug.zones_convex_hull || []).length} avg_area=${zoneAreaStats.avg.toFixed(2)}`
          : " n/a"}
      </div>
      {packTiming.count > 0 ? (
        <div className="zone-count">
          {`Pack avg=${((packTiming.avgMs || 0) / 1000).toFixed(2)}s last=${(
            (packTiming.lastMs || 0) / 1000
          ).toFixed(2)}s n=${packTiming.count}`}
        </div>
      ) : null}
      {packUiLog ? <div className="zone-count">{packUiLog}</div> : null}
      {zoneClickLogs.slice(-4).map((line, idx) => (
        <div className="zone-count" key={`zlog-${idx}`}>
          {line}
        </div>
      ))}
    </div>
  );

  const renderLoadingOverlay = (panel = "right") => {
    if (!sceneLoading) return null;
    if (isPackLoading && panel !== "right") return null;
    const elapsedSec = (packTiming.running ? packTiming.elapsedMs : 0) / 1000;
    const remainingSec = Math.max(0, 80 - elapsedSec);
    const avgSec =
      packTiming.count > 0 && Number.isFinite(packTiming.avgMs)
        ? (packTiming.avgMs / 1000).toFixed(2)
        : null;
    const lastSec =
      packTiming.lastMs != null && Number.isFinite(packTiming.lastMs)
        ? (packTiming.lastMs / 1000).toFixed(2)
        : null;
    return (
      <div className="loading-overlay">
        <div className="loading-overlay-card">
          <div className="loading-overlay-title">
            {packTiming.running ? `Packing... ${remainingSec.toFixed(1)}s` : "Loading..."}
          </div>
          {packTiming.running || avgSec || lastSec ? (
            <div className="loading-overlay-meta">
              {packTiming.running ? `Remaining ${remainingSec.toFixed(1)}s` : null}
              {avgSec ? `Avg ${avgSec}s` : null}
              {lastSec ? `Last ${lastSec}s` : null}
            </div>
          ) : null}
        </div>
      </div>
    );
  };

  const handleZoneRegionClick = (rid, evt) => {
    const source = zoneScene || scene;
    if (!source) return;
    const clickPt = getZoneStageWorldPoint(evt);
    const adjAll = buildRegionAdjacencyMulti(source.regions || [], [neighborSnap, 2]);
    const res = applyZoneDetachAttach(
      source,
      rid,
      adjAll,
      regionNeighborCursorRef.current
    );
    if (!res.changed) return;
    const currentAlias = getZoneAlias(res.currentZid, source);
    const neighborAliases = (res.neighborZids || []).map((zid) => getZoneAlias(zid, source));
    const clickLine = `Click: ${currentAlias} -> [${neighborAliases.join(", ")}]`;
    setZoneClickLogs((logs) => [...logs.slice(-20), clickLine]);
    setZoneScene(res.source);
    setSelectedZoneId(String(res.targetZid));
    saveZoneClickCache({
      x: clickPt?.x,
      y: clickPt?.y,
      rid,
      attach_to: res.targetZid,
    });
    refreshPackedFromZoneScene(res.source, enableBleed, selectedSource);
  };

  const centerRegionViewById = (zid) => {
    const source = regionZoneScene;
    if (!source?.zone_boundaries) return;
    const zidKey = String(zid);
    const zidNum = parseInt(zidKey, 10);
    const paths = source.zone_boundaries?.[zidKey] || source.zone_boundaries?.[zidNum] || [];
    if (!paths.length) return;
    const bounds = calcBounds(paths);
    const rect = region2WrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || region2StageSize.w;
    const viewH = rect?.height || region2StageSize.h;
    const bw = Math.max(1, bounds.maxx - bounds.minx);
    const bh = Math.max(1, bounds.maxy - bounds.miny);
    const fitScale = Math.max(
      0.4,
      Math.min(8, Math.min((viewW * 0.72) / bw, (viewH * 0.72) / bh))
    );
    const cx = (bounds.minx + bounds.maxx) / 2;
    const cy = (bounds.miny + bounds.maxy) / 2;
    setMainViewScale(fitScale);
    setMainViewPos({
      x: viewW / 2 - cx * fitScale,
      y: viewH / 2 - cy * fitScale,
    });
  };

  const centerPackedViewById = (zid) => {
    const source = packedSource || zoneScene || scene;
    if (!source?.zone_boundaries) return;
    const zidKey = String(zid);
    const zidNum = parseInt(zidKey, 10);
    const paths = source.zone_boundaries?.[zidKey] || source.zone_boundaries?.[zidNum] || [];
    if (!paths.length) return;
    const shift = source.zone_shift?.[zidKey] || source.zone_shift?.[zidNum] || [0, 0];
    const rot = source.zone_rot?.[zidKey] ?? source.zone_rot?.[zidNum] ?? 0;
    const center = source.zone_center?.[zidKey] || source.zone_center?.[zidNum] || [0, 0];
    const transformedPaths = paths.map((p) => transformPath(p, shift, rot, center));
    const bounds = calcBounds(transformedPaths);
    const bin = source?.placement_bin?.[zidKey] ?? source?.placement_bin?.[zidNum];
    const xOffset = bin === 1 ? (source?.canvas?.w || 0) + 40 : 0;
    const rect = regionWrapRef.current?.getBoundingClientRect();
    const viewW = rect?.width || regionStageSize.w;
    const viewH = rect?.height || regionStageSize.h;
    const bw = Math.max(1, bounds.maxx - bounds.minx);
    const bh = Math.max(1, bounds.maxy - bounds.miny);
    const fitScale = Math.max(
      0.25,
      Math.min(6, Math.min((viewW * 0.72) / bw, (viewH * 0.72) / bh))
    );
    const cx = (bounds.minx + bounds.maxx) / 2 + xOffset;
    const cy = (bounds.miny + bounds.maxy) / 2;
    setRegionScale(fitScale);
    setRegionPos({
      x: viewW / 2 - cx * fitScale,
      y: viewH / 2 - cy * fitScale,
    });
  };

  const handlePackedZoneSelect = (zid) => {
    const next = String(zid);
    setLeftTab("region");
    setSelectedZoneId(next);
    centerRegionViewById(next);
  };

  const getPackedStageWorldPoint = (evt) => {
    const stage = evt?.target?.getStage?.() || regionRef.current;
    if (!stage) return null;
    const pointer = stage.getPointerPosition?.();
    if (!pointer) return null;
    const scale = stage.scaleX?.() || regionScale || 1;
    return {
      x: (pointer.x - stage.x()) / scale,
      y: (pointer.y - stage.y()) / scale,
    };
  };

  const updateManualZoneTransform = (zid, nextDx, nextDy, nextRot, persist = false) => {
    const key = String(zid);
    setZoneScene((prev) => {
      if (!prev) return prev;
      const zone_shift = { ...(prev.zone_shift || {}) };
      const zone_rot = { ...(prev.zone_rot || {}) };
      if (Number.isFinite(nextDx) && Number.isFinite(nextDy)) zone_shift[key] = [nextDx, nextDy];
      if (Number.isFinite(nextRot)) zone_rot[key] = nextRot;
      return { ...prev, zone_shift, zone_rot };
    });
    setManualPackedEdits((prev) => {
      const next = { ...(prev || {}) };
      next[key] = {
        dx: Number.isFinite(nextDx) ? nextDx : prev?.[key]?.dx ?? 0,
        dy: Number.isFinite(nextDy) ? nextDy : prev?.[key]?.dy ?? 0,
        rot: Number.isFinite(nextRot) ? nextRot : prev?.[key]?.rot ?? 0,
      };
      if (persist) {
        const source = zoneScene || scene;
        if (source) scheduleSaveManualPackedEdits(next, source, selectedSource);
      }
      return next;
    });
  };

  const beginPackedEdit = (zid, evt) => {
    if (packedEditMode === "none") return;
    const source = packedSource || scene;
    if (!source) return;
    const key = String(zid);
    const shift =
      source.zone_shift?.[key] ||
      source.zone_shift?.[parseInt(key, 10)] ||
      [0, 0];
    const baseDx = Number.isFinite(shift?.[0]) ? shift[0] : 0;
    const baseDy = Number.isFinite(shift?.[1]) ? shift[1] : 0;
    const baseRot =
      source.zone_rot?.[key] ??
      source.zone_rot?.[parseInt(key, 10)] ??
      0;
    const center =
      source.zone_center?.[key] ||
      source.zone_center?.[parseInt(key, 10)] ||
      [0, 0];
    const pt = getPackedStageWorldPoint(evt);
    if (!pt) return;
    setSelectedZoneId(key);
    centerZoneViewById(key);
    if (packedEditMode === "move") {
      packedEditSessionRef.current = {
        type: "move",
        zid: key,
        startX: pt.x,
        startY: pt.y,
        baseDx,
        baseDy,
        baseRot,
      };
    } else if (packedEditMode === "rotate") {
      const cx = (center?.[0] || 0) + baseDx;
      const cy = (center?.[1] || 0) + baseDy;
      packedEditSessionRef.current = {
        type: "rotate",
        zid: key,
        cx,
        cy,
        startA: Math.atan2(pt.y - cy, pt.x - cx),
        baseDx,
        baseDy,
        baseRot,
      };
    }
    evt?.cancelBubble && (evt.cancelBubble = true);
  };

  const handlePackedStageMouseMove = (evt) => {
    const s = packedEditSessionRef.current;
    if (!s) return;
    const pt = getPackedStageWorldPoint(evt);
    if (!pt) return;
    if (s.type === "move") {
      const ndx = s.baseDx + (pt.x - s.startX);
      const ndy = s.baseDy + (pt.y - s.startY);
      updateManualZoneTransform(s.zid, ndx, ndy, s.baseRot, false);
    } else if (s.type === "rotate") {
      const a = Math.atan2(pt.y - s.cy, pt.x - s.cx);
      let d = ((a - s.startA) * 180) / Math.PI;
      while (d > 180) d -= 360;
      while (d < -180) d += 360;
      updateManualZoneTransform(s.zid, s.baseDx, s.baseDy, s.baseRot + d, false);
    }
  };

  const handlePackedStageMouseUp = () => {
    const s = packedEditSessionRef.current;
    if (!s) return;
    packedEditSessionRef.current = null;
    const source = zoneScene || scene;
    if (source) {
      const currentShift =
        source.zone_shift?.[s.zid] ||
        source.zone_shift?.[parseInt(s.zid, 10)] ||
        [s.baseDx, s.baseDy];
      const currentRot =
        source.zone_rot?.[s.zid] ??
        source.zone_rot?.[parseInt(s.zid, 10)] ??
        s.baseRot;
      updateManualZoneTransform(
        s.zid,
        Number.isFinite(currentShift?.[0]) ? currentShift[0] : s.baseDx,
        Number.isFinite(currentShift?.[1]) ? currentShift[1] : s.baseDy,
        Number.isFinite(currentRot) ? currentRot : s.baseRot,
        true
      );
    }
  };

  const transformOverlayToPacked = (item) => {
    const source = packedSource || scene;
    if (!source || !item || item.zid == null) return item;
    const zid = item.zid;
    const shift = source.zone_shift?.[zid] || source.zone_shift?.[parseInt(zid, 10)];
    const rot = source.zone_rot?.[zid] ?? source.zone_rot?.[parseInt(zid, 10)] ?? 0;
    const center =
      source.zone_center?.[zid] || source.zone_center?.[parseInt(zid, 10)] || [0, 0];
    const [pt] = transformPath([[item.x, item.y]], shift, rot, center);
    return {
      ...item,
      x: pt[0],
      y: pt[1],
      rotation: (item.rotation || 0) + (rot || 0),
    };
  };

  const saveState = async () => {
    if (!scene) return;
    await fetch(`/api/state?${sourceQuery(selectedSource)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        canvas: scene.canvas,
        regions: scene.regions,
        zone_boundaries: scene.zone_boundaries,
        svg_nodes: normalizeNodesForSave(nodes, sourceScale || 1),
        svg_segments: segs,
        source_voronoi: normalizeVoronoiForSave(sourceVoronoi, sourceScale || 1),
        labels,
        snap,
        view: {
          mainView: { scale: mainViewScale, pos: mainViewPos },
          region: { scale: regionScale, pos: regionPos },
        },
      }),
    });
  };

  const exportPdf = async () => {
    try {
      setError("");
      setExportMsg("");
      const exportStartTs = performance.now();
      setExportPdfLoading(true);
      setExportPdfTiming({ startTs: exportStartTs, elapsedMs: 0 });
      setExportPdfInfo(null);
      // Re-pack from latest source_region_scene before exporting.
      const sourceForExport = await getPackedSourceForCompute(selectedSource);
      await refreshPackedFromZoneScene(sourceForExport, enableBleed, selectedSource);
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
      await new Promise((resolve) => requestAnimationFrame(() => resolve()));
      const activeScene = sourceForExport || scene;
      if (!activeScene?.canvas) {
        throw new Error("canvas missing");
      }
      const size = { w: activeScene.canvas.w, h: activeScene.canvas.h };
      const buildRegionFillSvg = () => {
        const parts = [
          `<svg xmlns="http://www.w3.org/2000/svg" width="${size.w}" height="${size.h}" viewBox="0 0 ${size.w} ${size.h}">`,
        ];
        (activeScene.regions || []).forEach((poly, idx) => {
          if (!poly?.length) return;
          const d = `M ${poly.map((p) => `${p[0]} ${p[1]}`).join(" L ")} Z`;
          const fill = activeScene.region_colors?.[idx] || "#bbbbbb";
          parts.push(`<path d="${d}" fill="${escapeXml(fill)}"/>`);
        });
        parts.push("</svg>");
        return parts.join("");
      };
      const buildZoneOnlySvg = () => {
        const source = regionZoneScene || buildSnapZoneSceneFromRegionScene(activeScene);
        const parts = [
          `<svg xmlns="http://www.w3.org/2000/svg" width="${size.w}" height="${size.h}" viewBox="0 0 ${size.w} ${size.h}">`,
        ];
        const zoneOrder = source?.zone_order || [];
        const zonePolys = source?.zone_polys || [];
        if (zoneOrder.length && zonePolys.length) {
          zoneOrder.forEach((zidRaw, idx) => {
            const poly = zonePolys[idx];
            if (!poly?.length) return;
            const zid = String(zidRaw);
            const d = `M ${poly.map((p) => `${p[0]} ${p[1]}`).join(" L ")} Z`;
            parts.push(
              `<path d="${d}" fill="none" stroke="#000000" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>`
            );
            const lbl =
              source?.zone_labels?.[zid] ||
              source?.zone_labels?.[parseInt(zid, 10)];
            if (!lbl) return;
            const label =
              source?.zone_label_map?.[zid] ??
              source?.zone_label_map?.[parseInt(zid, 10)] ??
              lbl.label ??
              zid;
            parts.push(
              `<text x="${lbl.x}" y="${lbl.y}" fill="#000000" font-family="${escapeXml(
                labelFontFamily
              )}" font-size="${labelFontSize}" text-anchor="middle" dominant-baseline="middle">${escapeXml(
                label
              )}</text>`
            );
          });
        } else {
          Object.entries(source?.zone_boundaries || {}).forEach(([zid, paths]) => {
            (paths || []).forEach((poly) => {
              if (!poly?.length) return;
              const d = `M ${poly.map((p) => `${p[0]} ${p[1]}`).join(" L ")} Z`;
              parts.push(
                `<path d="${d}" fill="none" stroke="#000000" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>`
              );
            });
            const lbl =
              source?.zone_labels?.[zid] ||
              source?.zone_labels?.[parseInt(zid, 10)];
            if (!lbl) return;
            const label =
              source?.zone_label_map?.[zid] ??
              source?.zone_label_map?.[parseInt(zid, 10)] ??
              lbl.label ??
              zid;
            parts.push(
              `<text x="${lbl.x}" y="${lbl.y}" fill="#000000" font-family="${escapeXml(
                labelFontFamily
              )}" font-size="${labelFontSize}" text-anchor="middle" dominant-baseline="middle">${escapeXml(
                label
              )}</text>`
            );
          });
        }
        parts.push("</svg>");
        return parts.join("");
      };
      const pages = [
          {
            name: "region_fill_only",
            svg: buildRegionFillSvg(),
          },
          {
            name: "zone_index_only",
            svg: buildZoneOnlySvg(),
          },
          {
            name: "packed_image_nostroke",
            svg: captureStageSvg(regionRef, size, {
              "packed-image": true,
              "packed-overlay": true,
              "packed-stroke": false,
              "packed-label": true,
              "packed-hit": false,
            }),
          },
          {
            name: "packed_noimage_stroke_nolabel",
            svg: captureStageSvg(regionRef, size, {
              "packed-image": false,
              "packed-overlay": true,
              "packed-stroke": true,
              "packed-label": false,
              "packed-hit": false,
            }),
          },
      ];
      const res = await fetch(`/api/export_pdf?${sourceQuery(selectedSource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: selectedSource,
          pages,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
        const data = await res.json().catch(() => ({}));
        if (data?.name) {
          setExportPdfInfo({ name: data.name });
        }
        const htmlNames = [];
        try {
          const baseName = data?.name
            ? data.name.replace(/\.pdf$/i, "")
            : "convoi";
          const html0 = buildSimulateHtml(activeScene, packedLabels, labelFontFamily, labelFontSize);
          if (html0) {
            const name0 = `${baseName}_simulate.html`;
            await fetch("/api/save_html", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                name: name0,
                html: html0,
              }),
            });
            htmlNames.push(name0);
          }
        } catch {
          // ignore html export errors
        }
        setExportHtmlInfo(htmlNames);
        setExportMsg("Export PDF Done");
        setTimeout(() => setExportMsg(""), 3000);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setExportPdfLoading(false);
      setExportPdfTiming((prev) => ({ ...prev, elapsedMs: 0, startTs: 0 }));
    }
  };

  useEffect(() => {
    if (!simPlaying) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setSimProgress((p) => {
        const next = Math.min(1, p + dt / simTotalSeconds);
        if (next >= 1) setSimPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [simPlaying, simTotalSeconds]);

  useEffect(() => {
    if (!autoPack) return;
    const id = setTimeout(() => {
      repackCurrentPacked();
    }, 500);
    return () => clearTimeout(id);
  }, [
    packPadding,
    packMarginX,
    packMarginY,
    packBleed,
    enableBleed,
    packGrid,
    packAngle,
    packMode,
    autoPack,
    zoneScene,
    scene,
  ]);

  const parsePackedSvg = (text) => {
    const doc = new DOMParser().parseFromString(text, "image/svg+xml");
    const fill = doc.querySelector('g#fill');
    const bleed = doc.querySelector('g#bleed');
    const parsePaths = (node) =>
      Array.from(node?.querySelectorAll("path") || []).map((p) => ({
        d: p.getAttribute("d") || "",
        fill: p.getAttribute("fill") || "#000000",
        stroke: p.getAttribute("stroke") || "",
        strokeWidth: parseFloat(p.getAttribute("stroke-width") || "0") || 0,
      }));
    const fillPaths = parsePaths(fill).filter((p) => p.d);
    const bleedPaths = parsePaths(bleed).filter((p) => p.d);
    return { fillPaths, bleedPaths, hasBleed: !!bleed };
  };

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  const escapeXml = (value) => {
    if (value == null) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");
  };

  const matrixToAttr = (m) => {
    if (!m || m.length < 6) return "";
    const [a, b, c, d, e, f] = m.map((v) => (Number.isFinite(v) ? v : 0));
    return `matrix(${a} ${b} ${c} ${d} ${e} ${f})`;
  };

  const buildSvgFromStage = (stage, exportSize = null) => {
    const width = exportSize?.w || stage.width();
    const height = exportSize?.h || stage.height();
    const parts = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`,
    ];

    const pushAttrs = (attrs) => {
      const out = [];
      Object.entries(attrs).forEach(([key, val]) => {
        if (val == null || val === "" || val === false) return;
        out.push(`${key}="${escapeXml(val)}"`);
      });
      return out.join(" ");
    };

    const addShape = (node) => {
      if (!node.isVisible?.() || node.opacity?.() === 0) return;
      const transform = node.getAbsoluteTransform?.();
      const matrix = transform ? matrixToAttr(transform.getMatrix()) : "";
      const strokeScaleEnabled =
        typeof node.strokeScaleEnabled === "function" ? node.strokeScaleEnabled() : true;
      const common = {
        transform: matrix || undefined,
        opacity: node.opacity?.(),
        fill: node.fill?.() ?? undefined,
        "fill-opacity": node.fillOpacity?.(),
        stroke: node.stroke?.() ?? undefined,
        "stroke-opacity": node.strokeOpacity?.(),
        "stroke-width": node.strokeWidth?.(),
        "vector-effect": strokeScaleEnabled ? undefined : "non-scaling-stroke",
      };

      const className = node.getClassName?.();
      if (className === "Line") {
        const pts = node.points?.() || [];
        if (pts.length < 2) return;
        const pairs = [];
        for (let i = 0; i + 1 < pts.length; i += 2) {
          pairs.push(`${pts[i]},${pts[i + 1]}`);
        }
        const closed = node.closed?.();
        const tag = closed ? "polygon" : "polyline";
        const attrs = {
          ...common,
          points: pairs.join(" "),
          fill: closed ? common.fill ?? "none" : "none",
        };
        parts.push(`<${tag} ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Path") {
        const d = node.data?.();
        if (!d) return;
        const attrs = { ...common, d };
        parts.push(`<path ${pushAttrs(attrs)} />`);
        return;
      }

      if (className === "Rect") {
        const w = node.width?.();
        const h = node.height?.();
        if (!w || !h) return;
        const attrs = { ...common, x: 0, y: 0, width: w, height: h };
        parts.push(`<rect ${pushAttrs(attrs)} />`);
        return;
      }

        if (className === "Circle") {
          const r = node.radius?.();
          if (!r) return;
          const attrs = { ...common, cx: 0, cy: 0, r };
          parts.push(`<circle ${pushAttrs(attrs)} />`);
          return;
        }

        if (className === "Image") {
          const img = node.image?.();
          const src = img?.src;
          const w = node.width?.();
          const h = node.height?.();
          if (!src || !w || !h) return;
          const attrs = { ...common, x: 0, y: 0, width: w, height: h, href: src };
          parts.push(`<image ${pushAttrs(attrs)} />`);
          return;
        }

        if (className === "Text") {
          const text = node.text?.();
          if (text == null) return;
        const absPos = node.getAbsolutePosition?.() || { x: 0, y: 0 };
        const attrs = {
          fill: common.fill,
          "fill-opacity": common["fill-opacity"],
          stroke: common.stroke,
          "stroke-opacity": common["stroke-opacity"],
          "stroke-width": common["stroke-width"],
          opacity: common.opacity,
          x: absPos.x,
          y: absPos.y,
          "font-size": node.fontSize?.(),
          "font-family": node.fontFamily?.(),
          "text-anchor": node.align?.() === "center" ? "middle" : undefined,
          "dominant-baseline": "middle",
        };
        parts.push(`<text ${pushAttrs(attrs)}>${escapeXml(text)}</text>`);
      }
    };

    const walk = (node) => {
      const className = node.getClassName?.();
      if (className === "Group" || className === "Layer" || className === "Stage") {
        const children = node.getChildren?.() || [];
        children.forEach((child) => walk(child));
        return;
      }
      addShape(node);
    };

    walk(stage);
    parts.push("</svg>");
    return parts.join("");
  };

  const handleSimVideoDownload = async () => {
    if (simVideoLoading || !scene) return;
    setSimVideoLoading(true);
    try {
      const res = await fetch(`/api/export_sim_video?${sourceQuery(selectedSource)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source: selectedSource,
          scene,
          packedLabels,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data?.name) {
        window.location = `/api/download_sim_video?name=${encodeURIComponent(data.name)}`;
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSimVideoLoading(false);
    }
  };

  const handleSimPlayToggle = () => {
    if (!simPlaying && simProgress >= 1) {
      setSimProgress(0);
    }
    setSimPlaying((v) => !v);
  };

  const injectSvgLabels = (svgText, labels, fontFamily, fontSize) => {
    try {
      const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
      const svg = doc.querySelector("svg");
      if (!svg) return svgText;
      Array.from(svg.querySelectorAll("text")).forEach((n) => n.remove());
      Object.values(labels || {}).forEach((lbl) => {
        const x = Number(lbl.x);
        const y = Number(lbl.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return;
        const text = doc.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", String(x));
        text.setAttribute("y", String(y));
        text.setAttribute("fill", "#ffffff");
        text.setAttribute("font-family", fontFamily || "Arial");
        text.setAttribute("font-size", String(fontSize || 12));
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("dominant-baseline", "middle");
        text.textContent = String(lbl.label ?? "");
        svg.appendChild(text);
      });
      return new XMLSerializer().serializeToString(svg);
    } catch {
      return svgText;
    }
  };

  const captureStageSvg = (ref, exportSize = null, layerVisibility = null) => {
    const stage = ref?.current;
    if (!stage) return "";
    const prevScale = stage.scale();
    const prevPos = stage.position();
    const prevVis = [];
    const applyVis = (name, visible) => {
      let nodes = stage.find(`.${name}`) || [];
      let list = nodes?.toArray ? nodes.toArray() : nodes;
      if (!list || list.length === 0) {
        nodes = stage.find((n) => (n.name && n.name() === name) || false) || [];
        list = nodes?.toArray ? nodes.toArray() : nodes;
      }
      (list || []).forEach((n) => {
        prevVis.push([n, n.visible()]);
        n.visible(visible);
      });
    };
    if (layerVisibility) {
      Object.entries(layerVisibility).forEach(([name, visible]) => applyVis(name, visible));
    }
    stage.scale({ x: 1, y: 1 });
    stage.position({ x: 0, y: 0 });
    stage.draw();
    const svg =
      typeof stage.toSVG === "function"
        ? stage.toSVG()
        : buildSvgFromStage(stage, exportSize);
    prevVis.forEach(([node, vis]) => node.visible(vis));
    stage.scale(prevScale);
    stage.position(prevPos);
    stage.draw();
    return svg;
  };

  const downloadStage = (ref, filename, exportSize = null) => {
    try {
      const svg = captureStageSvg(ref, exportSize);
      try {
        let suffix = "";
        if (showImages) suffix += "_image";
        if (showStroke) suffix += "_stroke";
        suffix += showLabels ? "_label" : "_nolabel";
        const nameWithSuffix = filename.replace(/\.svg$/i, `${suffix}.svg`);
        fetch("/api/save_konva_svg", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: nameWithSuffix, svg }),
        });
      } catch {
        // ignore save errors
      }
      // backend save only
    } catch {
      // ignore download errors
    }
  };

  useEffect(() => {
    if (zoneScene) return;
    if (!packedImageSrc) return;
    fetch(packedImageSrc)
      .then((res) => res.text())
      .then((text) => {
        const parsed = parsePackedSvg(text);
        setPackedFillPaths(parsed.fillPaths);
        setPackedBleedPaths(parsed.bleedPaths);
        if (!parsed.hasBleed) {
          setPackedBleedError("packed.svg missing bleed layer");
        } else {
          setPackedBleedError("");
        }
      })
      .catch(() => {
        setPackedFillPaths([]);
        setPackedBleedPaths([]);
        setPackedBleedError("packed.svg failed to load");
      });
  }, [packedImageSrc]);

  useEffect(() => {
    if (zoneScene) return;
    if (!packedImageSrc2) return;
    fetch(packedImageSrc2)
      .then((res) => res.text())
      .then((text) => {
        const parsed = parsePackedSvg(text);
        setPackedFillPaths2(parsed.fillPaths);
        setPackedBleedPaths2(parsed.bleedPaths);
        if (!parsed.hasBleed) {
          setPackedBleedError2("packed_page2.svg missing bleed layer");
        } else {
          setPackedBleedError2("");
        }
      })
      .catch(() => {
        setPackedFillPaths2([]);
        setPackedBleedPaths2([]);
        setPackedBleedError2("packed_page2.svg failed to load");
      });
  }, [packedImageSrc2]);

  const nodeLayer = useMemo(() => {
    if (!segs.length || !nodes.length) return null;
    return segs.map(([a, b], idx) => {
      const p1 = nodes[a];
      const p2 = nodes[b];
      return (
        <Line
          key={`s-${idx}`}
          points={[p1.x, p1.y, p2.x, p2.y]}
          stroke="#f5f6ff"
          opacity={0.45}
          strokeWidth={0.5 / mainViewScale}
          strokeScaleEnabled={false}
        />
      );
    });
  }, [segs, nodes, mainViewScale]);

  const borderLayer = useMemo(() => {
    if (!borderSegments.length) return null;
    return borderSegments.map((seg, idx) => (
      <Line
        key={`b-${idx}`}
        points={toPoints(seg)}
        stroke="#2f80ff"
        opacity={0.45}
        strokeWidth={0.5 / mainViewScale}
        strokeScaleEnabled={false}
      />
    ));
  }, [borderSegments, mainViewScale]);

  const sourceVoronoiLayer = useMemo(() => {
    if (!sourceVoronoi?.cells?.length) return null;
    return sourceVoronoi.cells.map((poly, idx) => (
      <Line
        key={`sv-${idx}`}
        points={toPoints(poly)}
        closed
        stroke="#ff5a36"
        fill="rgba(255,90,54,0.22)"
        opacity={0.9}
        dash={[4 / mainViewScale, 2 / mainViewScale]}
        strokeWidth={1.25 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
    ));
  }, [sourceVoronoi, mainViewScale]);

  const sourceSnapLayer = useMemo(() => {
    if (!sourceVoronoiGraph?.refs?.length || !sourceVoronoiGraph?.vertices?.length) return null;
    return sourceVoronoiGraph.refs.map((polyRefs, idx) => {
      const poly = (polyRefs || [])
        .map((vid) => sourceVoronoiGraph.vertices[vid])
        .filter(Boolean)
        .map((v) => [v.x, v.y]);
      if (poly.length < 3) return null;
      return (
      <Line
        key={`ss-${idx}`}
        points={toPoints(poly)}
        closed
        stroke="#00d26a"
        fill="rgba(0,210,106,0.38)"
        opacity={1}
        strokeWidth={2 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
      );
    });
  }, [sourceVoronoiGraph, mainViewScale]);

  const sourceMaskBorderLayer = useMemo(() => {
    if (!sourceVoronoi?.mask?.length) return null;
    return (
      <Line
        key="source-mask-border"
        points={toPoints(sourceVoronoi.mask)}
        closed
        stroke="#2f80ff"
        opacity={1}
        strokeWidth={2.5 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
    );
  }, [sourceVoronoi, mainViewScale]);

  const sourceMaskFillLayer = useMemo(() => {
    if (!sourceVoronoi?.mask?.length) return null;
    return (
      <Line
        key="source-mask-fill"
        points={toPoints(sourceVoronoi.mask)}
        closed
        fill="rgba(0,210,106,0.18)"
        opacity={1}
        listening={false}
      />
    );
  }, [sourceVoronoi]);

  const sourceVoronoiDebugLayer = useMemo(() => {
    const rawCount = sourceVoronoi?.cells?.length || 0;
    const snapCount = sourceVoronoi?.snappedCells?.length || 0;
    return (
      <Text
        x={12}
        y={12}
        text={`voronoi:${rawCount} snap:${snapCount}`}
        fontSize={18 / mainViewScale}
        fill="#7CFFB2"
        stroke="#00112b"
        strokeWidth={3 / mainViewScale}
        listening={false}
      />
    );
  }, [sourceVoronoi, mainViewScale]);

  const sourceVoronoiVertexLayer = useMemo(() => {
    if (!sourceVoronoiGraph?.vertices?.length) return null;
    return sourceVoronoiGraph.vertices.map((v) => (
      <Circle
        key={`svh-${v.id}`}
        x={v.x}
        y={v.y}
        radius={3.5 / mainViewScale}
        fill="#00d26a"
        stroke="#ffffff"
        strokeWidth={1 / mainViewScale}
        strokeScaleEnabled={false}
        draggable={!edgeMode && !deleteEdgeMode && !addNodeMode}
        onDragStart={() => {
          sourceDragSnapshotRef.current = createSourceEditSnapshot();
        }}
        onDragMove={(e) => {
          const target = resolveSourceSnapTarget(e.target.x(), e.target.y(), {
            excludeVoronoiId: v.id,
            includeNodes: true,
            includeVoronoi: false,
            radius: greenToRedSnapRadius,
          });
          const nextTarget =
            target?.kind === "none"
              ? { x: e.target.x(), y: e.target.y(), kind: "none", id: null }
              : target;
          updateSourceVoronoiVertex(v.id, nextTarget.x, nextTarget.y, {
            snapTarget: nextTarget,
          });
          e.target.x(nextTarget.x);
          e.target.y(nextTarget.y);
        }}
        onDragEnd={(e) => {
          const target = resolveSourceSnapTarget(e.target.x(), e.target.y(), {
            excludeVoronoiId: v.id,
            includeNodes: true,
            includeVoronoi: false,
            radius: greenToRedSnapRadius,
          });
          const finalTarget =
            target?.kind === "none"
              ? { x: e.target.x(), y: e.target.y(), kind: "none", id: null }
              : target;
          const currentVoronoi = sourceVoronoi || { mask: [], cells: [], snappedCells: [] };
          const graph = buildVoronoiVertexGraph(currentVoronoi?.snappedCells || []);
          let nextVoronoi = currentVoronoi;
          if (graph.vertices[v.id]) {
            const nextVertices = graph.vertices.map((item) =>
              item.id === v.id ? { ...item, x: finalTarget.x, y: finalTarget.y } : { ...item }
            );
            nextVoronoi = {
              ...currentVoronoi,
              snappedCells: rebuildVoronoiCellsFromGraph(nextVertices, graph.refs),
            };
          }
          updateSourceVoronoiVertex(v.id, finalTarget.x, finalTarget.y, { snapTarget: finalTarget });
          e.target.x(finalTarget.x);
          e.target.y(finalTarget.y);
          commitSourceDragHistory(nodes, segs, nextVoronoi);
          void persistSourceCacheNow(nodes, segs, nextVoronoi, selectedSource);
        }}
      />
    ));
  }, [
    sourceVoronoiGraph,
    mainViewScale,
    edgeMode,
    deleteEdgeMode,
    addNodeMode,
    commitSourceDragHistory,
    createSourceEditSnapshot,
    greenToRedSnapRadius,
    nodes,
    persistSourceCacheNow,
    resolveSourceSnapTarget,
    segs,
    selectedSource,
    sourceVoronoi,
    updateSourceVoronoiVertex,
  ]);

  const regionSnapOverlayLayer = useMemo(() => {
    const snappedCells = scene?.voronoi?.snapped_cells || [];
    if (!snappedCells.length) return null;
    return snappedCells.map((poly, idx) => (
      <Line
        key={`region-snap-${idx}`}
        points={toPoints(poly)}
        closed
        stroke="#ff2d55"
        opacity={1}
        strokeWidth={1 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
    ));
  }, [scene, mainViewScale]);

  const regionZoneScene = useMemo(() => buildSnapZoneSceneFromRegionScene(scene), [scene]);

  const zoneVerticesLayer = useMemo(() => {
    const radius = 10 / Math.max(mainViewScale, 0.1);
    return Object.entries(zoneVertices || {}).flatMap(([zid, pts]) =>
      (Array.isArray(pts) ? pts : []).map((pt, idx) => (
        <Circle
          key={`zone-vert-${zid}-${idx}`}
          x={pt[0]}
          y={pt[1]}
          radius={radius}
          fill="#ff3b30"
          stroke="#ff3b30"
          strokeWidth={1 / Math.max(mainViewScale, 0.1)}
          listening={false}
          opacity={0.9}
        />
      ))
    );
  }, [zoneVertices, mainViewScale]);

  const regionZoneIndexLayer = useMemo(() => {
    const snapMap = scene?.snap_region_map || {};
    const regions = scene?.regions || [];
    const items = Object.entries(snapMap).map(([zid, regionIds]) => {
      let sumArea = 0;
      let sumX = 0;
      let sumY = 0;
      (regionIds || []).forEach((ridRaw) => {
        const rid = parseInt(ridRaw, 10);
        const poly = regions[rid];
        if (!poly?.length) return;
        const { area, cx, cy } = polyCentroid(poly);
        const w = Math.abs(area) || 1;
        sumArea += w;
        sumX += cx * w;
        sumY += cy * w;
      });
      if (sumArea <= 0) return null;
      return {
        zid: String(zid),
        x: sumX / sumArea,
        y: sumY / sumArea,
      };
    }).filter(Boolean);
    if (!items.length) return null;
    return items.map((item) => {
      const text = `${parseInt(item.zid, 10) + 1}`;
      const size = Math.max(labelFontSize / mainViewScale, 6 / mainViewScale);
      const metrics = measureText(text, size, labelFontFamily);
      const isSelected = String(item.zid) === String(selectedZoneId);
      return (
        <Text
          key={`region-zone-label-${item.zid}`}
          x={item.x}
          y={item.y}
          text={text}
          fill={isSelected ? "#ff3b30" : "#ffffff"}
          stroke="rgba(0,0,0,0.5)"
          strokeWidth={1 / mainViewScale}
          fontSize={size}
          fontFamily={labelFontFamily}
          fontStyle="normal"
          align="center"
          verticalAlign="middle"
          offsetX={metrics.width / 2}
          offsetY={metrics.height / 2}
          perfectDrawEnabled={false}
          listening
          hitStrokeWidth={10 / mainViewScale}
          onClick={() => {
            setSelectedZoneId(String(item.zid));
            centerPackedViewById(String(item.zid));
          }}
          onTap={() => {
            setSelectedZoneId(String(item.zid));
            centerPackedViewById(String(item.zid));
          }}
          onMouseDown={() => {
            setSelectedZoneId(String(item.zid));
            centerPackedViewById(String(item.zid));
          }}
          onTouchStart={() => {
            setSelectedZoneId(String(item.zid));
            centerPackedViewById(String(item.zid));
          }}
        />
      );
    });
  }, [scene, mainViewScale, labelFontFamily, labelFontSize, selectedZoneId]);

  const regionSelectedZoneOutlineLayer = useMemo(() => {
    if (!regionZoneScene?.zone_boundaries || selectedZoneId == null) return null;
    const paths =
      regionZoneScene.zone_boundaries?.[selectedZoneId] ||
      regionZoneScene.zone_boundaries?.[parseInt(selectedZoneId, 10)] ||
      [];
    return (paths || []).map((p, idx) => (
      <Line
        key={`region-zone-selected-${idx}`}
        points={toPoints(p)}
        closed
        stroke="#ffffff"
        strokeWidth={20 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
    ));
  }, [regionZoneScene, selectedZoneId, mainViewScale]);

  const zoneOutlineLayer = useMemo(() => {
    if (!borderSegments.length) return null;
    return borderSegments.map((seg, idx) => (
      <Line
        key={`zb-outline-${idx}`}
        points={toPoints(seg)}
        stroke="#8b1e24"
        strokeWidth={3 / mainViewScale}
        strokeScaleEnabled={false}
        listening={false}
      />
    ));
  }, [borderSegments, mainViewScale]);

  function offsetPoints(pts, dx, dy) {
    return (pts || []).map((p) => [p[0] + dx, p[1] + dy]);
  }

  const zoneColorMap = useMemo(() => {
    if (!scene?.region_colors || !scene?.zone_id) return {};
    const map = {};
    for (let i = 0; i < scene.zone_id.length; i++) {
      const zid = scene.zone_id[i];
      if (map[zid]) continue;
      const color = scene.region_colors[i];
      if (color) map[zid] = color;
    }
    return map;
  }, [scene]);

  useEffect(() => {
    if (autoFit && (scene?.canvas || scene?.regions?.length)) {
      if (leftTab === 'region') {
        fitMainViewToView(calcBounds(scene.regions || []));
      } else {
        if (scene?.canvas) {
          fitMainViewToView({ minx: 0, miny: 0, maxx: scene.canvas.w, maxy: scene.canvas.h });
        } else {
          fitMainViewToView(calcBounds(scene.regions || []));
        }
      }

      if (scene?.canvas) {
        fitRegionToView({ minx: 0, miny: 0, maxx: scene.canvas.w, maxy: scene.canvas.h });
      } else {
        fitRegionToView(calcBounds(scene.regions || []));
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, stageSize, regionStageSize, region2StageSize, zoneStageSize, autoFit, leftTab]);

  return (
    <div className="app">
      <div className="content">
        <div className="column-left">
          <div className="panel toolbar pack-toolbar-right">
            <button onClick={exportPdf}>Export PDF</button>
            <button
              onClick={() => {
                setSimProgress(0);
                setSimPlaying(false);
                setShowSim(true);
              }}
            >
              Simulate
            </button>
            <label className="mini-input">
              Recent files
              <select
                value={selectedSource}
                onChange={(e) => {
                  const next = e.target.value;
                  setSelectedSource(next);
                  loadScene(
                    true,
                    sourceOnlyMode ? false : leftTab !== "source",
                    sourceOnlyMode ? false : leftTab !== "source",
                    false,
                    next
                  );
                }}
              >
                {recentFiles.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <div className="toolbar-spacer" />
            {exportMsg ? <div className="meta">{exportMsg}</div> : null}
            {error ? <div className="error">{error}</div> : null}
          </div>

          <div className="view-tabs-row">
            <div className="panel view-tabs">
              <button className={leftTab === 'source' ? 'active' : ''} onClick={() => setLeftTab('source')}>Source</button>
              <button className={leftTab === 'region' ? 'active' : ''} onClick={() => setLeftTab('region')}>Region + Zone</button>
            </div>
            <div className="view-tabs-scale">
              <label htmlFor="source-scale">Scale</label>
              <input
                id="source-scale"
                type="number"
                min="0.1"
                max="5"
                step="0.01"
                value={sourceScale}
                onChange={(e) => handleSourceScaleChange(e.target.value)}
              />
            </div>
          </div>

          {leftTab === 'source' && (
            <div className={`left ${leftPanelLoading ? "is-loading" : ""}`} ref={leftRef}>
              <div className="preview-header">
                <div className="preview-title">Source (Konva)</div>
                <div className="preview-controls">
                  <button
                    className="btn"
                    onClick={() => saveSourceEditsAndRefreshRegion(selectedSource)}
                  >
                    Export
                  </button>
                  <button
                    className="btn"
                    onClick={() => resetSourceCacheAndReload(selectedSource)}
                  >
                    Reset
                  </button>
                  <button
                    className="icon-button"
                    title="Undo"
                    disabled={!sourceUndoStack.length}
                    onClick={undoSourceEdit}
                  >
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <path
                        d="M8 5 L3 10 L8 15"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M4 10 H11 C14.3 10 17 12.2 17 15"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                  <button
                    className="icon-button"
                    title="Redo"
                    disabled={!sourceRedoStack.length}
                    onClick={redoSourceEdit}
                  >
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <path
                        d="M12 5 L17 10 L12 15"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                      <path
                        d="M16 10 H9 C5.7 10 3 12.2 3 15"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    </svg>
                  </button>
                  <button
                    className={`icon-button ${edgeMode ? "active" : ""}`}
                    title="Create Edge"
                    onClick={() => {
                      setEdgeMode((v) => !v);
                      setAddNodeMode(false);
                      setDeleteEdgeMode(false);
                      setEdgeCandidate(null);
                      setDeleteEdgeCandidate(null);
                    }}
                  >
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <circle cx="4" cy="4" r="2" fill="currentColor" />
                      <circle cx="16" cy="16" r="2" fill="currentColor" />
                      <line x1="5.5" y1="5.5" x2="14.5" y2="14.5" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </button>
                  <button
                    className={`icon-button ${addNodeMode ? "active" : ""}`}
                    title="Add Node"
                    onClick={() => {
                      setAddNodeMode((v) => !v);
                      setEdgeMode(false);
                      setDeleteEdgeMode(false);
                      setEdgeCandidate(null);
                      setDeleteEdgeCandidate(null);
                    }}
                  >
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <circle cx="10" cy="10" r="3" fill="currentColor" />
                      <line x1="10" y1="4" x2="10" y2="16" stroke="currentColor" strokeWidth="2" />
                      <line x1="4" y1="10" x2="16" y2="10" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </button>
                  <button
                    className={`icon-button ${deleteEdgeMode ? "active" : ""}`}
                    title="Delete Edge"
                    onClick={() => {
                      setDeleteEdgeMode((v) => !v);
                      setEdgeMode(false);
                      setAddNodeMode(false);
                      setEdgeCandidate(null);
                      setDeleteEdgeCandidate(null);
                    }}
                  >
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <line x1="4" y1="4" x2="16" y2="16" stroke="currentColor" strokeWidth="2" />
                      <line x1="16" y1="4" x2="4" y2="16" stroke="currentColor" strokeWidth="2" />
                    </svg>
                  </button>
                  <button className="icon-button" title="Overlay" onClick={handleOverlayPick}>
                    <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                      <rect x="3" y="4" width="12" height="10" rx="1" ry="1" stroke="currentColor" strokeWidth="2" fill="none" />
                      <rect x="7" y="6" width="10" height="10" rx="1" ry="1" stroke="currentColor" strokeWidth="2" fill="none" opacity="0.7" />
                    </svg>
                  </button>
                  <input
                    ref={overlayInputRef}
                    type="file"
                    accept=".svg"
                    style={{ display: "none" }}
                    onChange={handleOverlayFileChange}
                  />
                  <label className="mini-input">
                    Overlay Fill
                    <input
                      type="color"
                      value={overlayFill}
                      onChange={(e) => {
                        const color = e.target.value;
                        setOverlayFill(color);
                        if (selectedOverlayId) updateOverlayColor(selectedOverlayId, color);
                      }}
                    />
                  </label>
                  <button
                    className="icon-button"
                    title="Download"
                    onClick={() =>
                      downloadStage(stageRef, "source-konva.svg", scene?.canvas || null)
                  }
                  >
                    {"\u2193"}
                  </button>
                </div>
              </div>
              <Stage
                width={stageSize.w}
                height={stageSize.h}
                draggable
                scaleX={mainViewScale}
                scaleY={mainViewScale}
                x={mainViewPos.x}
                y={mainViewPos.y}
                onWheel={handleMainViewWheel}
                onDragMove={handleMainViewDragMove}
                onDragEnd={handleMainViewDragMove}
                onMouseMove={() => {
                    const stage = stageRef.current;
                    const pointer = stage.getPointerPosition();
                    if (!pointer) return;
                    const world = {
                      x: (pointer.x - mainViewPos.x) / mainViewScale,
                      y: (pointer.y - mainViewPos.y) / mainViewScale,
                    };
                    if (edgeMode) {
                      const cand = findEdgeCandidate(world);
                      setEdgeCandidate(cand);
                      setDeleteEdgeCandidate(null);
                    } else if (deleteEdgeMode) {
                      const cand = findExistingEdgeCandidate(world);
                      setDeleteEdgeCandidate(cand);
                      setEdgeCandidate(null);
                    } else {
                      setEdgeCandidate(null);
                      setDeleteEdgeCandidate(null);
                    }
                  }}
                  onMouseLeave={() => {
                    if (edgeMode) setEdgeCandidate(null);
                    if (deleteEdgeMode) setDeleteEdgeCandidate(null);
                  }}
                  onMouseDown={() => {
                    const stage = stageRef.current;
                    const pointer = stage.getPointerPosition();
                    if (!pointer) return;
                    const world = {
                      x: (pointer.x - mainViewPos.x) / mainViewScale,
                      y: (pointer.y - mainViewPos.y) / mainViewScale,
                    };
                    if (edgeMode) {
                      if (!edgeCandidate) return;
                      const key = edgeKey(edgeCandidate.a, edgeCandidate.b);
                      const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
                      if (segSet.has(key)) return;
                      pushSourceUndoSnapshot(createSourceEditSnapshot());
                      const nextSegs = [...segs, [edgeCandidate.a, edgeCandidate.b]];
                      setSegs(nextSegs);
                      return;
                    }
                    if (deleteEdgeMode) {
                      if (!deleteEdgeCandidate) return;
                      pushSourceUndoSnapshot(createSourceEditSnapshot());
                      const nextSegs = segs.filter((_, idx) => idx !== deleteEdgeCandidate.idx);
                      const pruned = pruneIsolatedNodes(nodes, nextSegs);
                      setNodes(pruned.nodes);
                      setSegs(pruned.segs);
                      return;
                    }
                    if (addNodeMode) {
                      pushSourceUndoSnapshot(createSourceEditSnapshot());
                      const nextNodes = [...nodes, { id: nodes.length, x: world.x, y: world.y }];
                      let nextSegs = [...segs];
                      if (nodes.length) {
                        let nearest = 0;
                        let best = Infinity;
                        nodes.forEach((n, idx) => {
                          const d = Math.hypot(n.x - world.x, n.y - world.y);
                          if (d < best) {
                            best = d;
                            nearest = idx;
                          }
                        });
                        nextSegs.push([nextNodes.length - 1, nearest]);
                      }
                      setNodes(nextNodes);
                      setSegs(nextSegs);
                    }
                  }}
                  ref={stageRef}
                ><Layer>{scene?.canvas ? (
                <Rect
                  x={0}
                  y={0}
                  width={scene.canvas.w}
                  height={scene.canvas.h}
                  stroke="#ffffff"
                  strokeWidth={0.5 / mainViewScale}
                  listening={false}
                />
              ) : null}</Layer><Layer>{nodeLayer}</Layer><Layer>{borderLayer}</Layer>{edgeCandidate ? (
                <Layer><Line
                    points={[
                      nodes[edgeCandidate.a].x,
                      nodes[edgeCandidate.a].y,
                      nodes[edgeCandidate.b].x,
                      nodes[edgeCandidate.b].y,
                    ]}
                  stroke="#cfd6ff"
                  opacity={0.4}
                    strokeWidth={0.5 / mainViewScale}
                    strokeScaleEnabled={false}
                  /></Layer>
              ) : null}{deleteEdgeCandidate ? (
                <Layer><Line
                    points={[
                      nodes[deleteEdgeCandidate.a].x,
                      nodes[deleteEdgeCandidate.a].y,
                      nodes[deleteEdgeCandidate.b].x,
                      nodes[deleteEdgeCandidate.b].y,
                    ]}
                    stroke="#ff3b30"
                    opacity={0.6}
                    strokeWidth={0.5 / mainViewScale}
                    strokeScaleEnabled={false}
                  /></Layer>
              ) : null}<Layer name="source-overlay">{overlayItems.map((item) =>
                  item.img ? (
                    <Image
                      key={item.id}
                      image={item.img}
                      x={item.x}
                      y={item.y}
                      width={item.width}
                      height={item.height}
                      offsetX={item.width / 2}
                      offsetY={item.height / 2}
                      scaleX={item.scaleX}
                      scaleY={item.scaleY}
                      rotation={item.rotation}
                      draggable={!edgeMode && !deleteEdgeMode && !addNodeMode}
                      onClick={() => setSelectedOverlayId(item.id)}
                      onTap={() => setSelectedOverlayId(item.id)}
                      onDragEnd={(e) => {
                        const nx = e.target.x();
                        const ny = e.target.y();
                        const zid = findZoneAtPoint({ x: nx, y: ny });
                        const next = overlayItems.map((o) =>
                          o.id === item.id ? { ...o, x: nx, y: ny, zid } : o
                        );
                        setOverlayItems(next);
                      }}
                      onTransformEnd={() => {
                        const node = overlayNodeRefs.current[item.id];
                        if (!node) return;
                        const scaleX = node.scaleX();
                        const scaleY = node.scaleY();
                        const rotation = node.rotation();
                        const nx = node.x();
                        const ny = node.y();
                        node.scaleX(1);
                        node.scaleY(1);
                        const next = overlayItems.map((o) =>
                          o.id === item.id
                            ? {
                                ...o,
                                x: nx,
                                y: ny,
                                rotation,
                                scaleX: o.scaleX * scaleX,
                                scaleY: o.scaleY * scaleY,
                              }
                            : o
                        );
                        setOverlayItems(next);
                      }}
                      ref={(node) => {
                        if (node) overlayNodeRefs.current[item.id] = node;
                      }}
                    />
                  ) : null
                )}<Transformer
                  ref={overlayTransformerRef}
                  rotateEnabled
                  enabledAnchors={[
                    "top-left",
                    "top-right",
                    "bottom-left",
                    "bottom-right",
                  ]}
                  boundBoxFunc={(oldBox, newBox) => {
                    if (newBox.width < 10 || newBox.height < 10) return oldBox;
                    return newBox;
                  }}
                /></Layer><Layer>{nodes.map((n) => (
                  <Circle
                    key={`n-${n.id}`}
                    x={n.x}
                    y={n.y}
                    radius={3 / mainViewScale}
                    fill="red"
                    strokeScaleEnabled={false}
                    draggable={!edgeMode && !deleteEdgeMode && !addNodeMode}
                    onDragStart={() => {
                      sourceDragSnapshotRef.current = createSourceEditSnapshot();
                    }}
                    onDragMove={(e) => {
                      const target = resolveSourceSnapTarget(e.target.x(), e.target.y(), {
                        excludeNodeId: n.id,
                        includeNodes: true,
                        includeVoronoi: false,
                        radius: liveSnapRadius,
                      });
                      e.target.x(target.x);
                      e.target.y(target.y);
                      const next = nodes.map((p) =>
                        p.id === n.id ? { ...p, x: target.x, y: target.y } : p
                      );
                      setNodes(next);
                    }}
                    onDragEnd={(e) => {
                      const target = resolveSourceSnapTarget(e.target.x(), e.target.y(), {
                        excludeNodeId: n.id,
                        includeNodes: true,
                        includeVoronoi: false,
                        radius: liveSnapRadius,
                      });
                      e.target.x(target.x);
                      e.target.y(target.y);
                      const next = nodes.map((p) =>
                        p.id === n.id ? { ...p, x: target.x, y: target.y } : p
                      );
                      setNodes(next);
                      commitSourceDragHistory(next, segs, sourceVoronoi);
                      void persistSourceCacheNow(next, segs, sourceVoronoi, selectedSource);
                    }}
                  />
              ))}</Layer><Layer name="source-mask-fill-top">{sourceMaskFillLayer}</Layer><Layer name="source-voronoi-top">{sourceVoronoiLayer}</Layer><Layer name="source-snap-top">{sourceSnapLayer}</Layer><Layer name="source-mask-border-top">{sourceMaskBorderLayer}</Layer><Layer name="source-voronoi-debug">{sourceVoronoiDebugLayer}</Layer><Layer name="source-voronoi-vertices">{sourceVoronoiVertexLayer}</Layer></Stage>
              {renderLoadingOverlay("left")}
              {renderLeftDebug()}
            </div>
          )}

          {leftTab === 'region' && (
            <div className={`preview half ${leftPanelLoading ? "is-loading" : ""}`} ref={region2WrapRef}>
              <div className="preview-header">
                <div className="preview-title">Region + Zone (Konva)</div>
                <div className="preview-controls">
                  <button
                    className="btn"
                    onClick={() => {
                      packFromRegionSnap();
                    }}
                    disabled={computeBusy}
                  >
                    {computeBusy ? "Packing..." : "Pack"}
                  </button>
                  <button
                    className="icon-button"
                    title="Download"
                    onClick={() =>
                      downloadStage(region2Ref, "region-konva.svg", scene?.canvas || null)
                    }
                  >
                    {"\u2193"}
                  </button>
                </div>
              </div>
              {scene ? (
                <Stage
                  width={region2StageSize.w}
                  height={region2StageSize.h}
                  draggable
                  scaleX={mainViewScale}
                  scaleY={mainViewScale}
                  x={mainViewPos.x}
                  y={mainViewPos.y}
                  onWheel={handleMainViewWheel}
                  onDragMove={handleMainViewDragMove}
                  onDragEnd={handleMainViewDragMove}
                  ref={region2Ref}
                  ><Layer>{scene?.canvas ? (
                      <Rect
                        x={0}
                        y={0}
                        width={scene.canvas.w}
                        height={scene.canvas.h}
                        stroke="#ffffff"
                        strokeWidth={2 / mainViewScale}
                        listening={false}
                      />
                    ) : null}</Layer><Layer>{scene.regions.map((poly, idx) => (
                      <Line
                        key={`r2-${idx}`}
                        points={toPoints(poly)}
                        closed
                        stroke="#ffffff"
                        fill={scene.region_colors?.[idx] || "#bbb"}
                        strokeWidth={1 / mainViewScale}
                        strokeScaleEnabled={false}
                      />
                    ))}</Layer><Layer name="zone-vertices">{zoneVerticesLayer}</Layer><Layer name="region-zone-index">{regionZoneIndexLayer}</Layer><Layer name="region-snap-overlay">{regionSnapOverlayLayer}</Layer><Layer name="region-zone-selected">{regionSelectedZoneOutlineLayer}</Layer></Stage>
              ) : null}
              {renderLoadingOverlay("left")}
              {renderLeftDebug()}
            </div>
          )}
        </div>
        <div className="right">
          {!sourceOnlyMode ? (
          <div className="panel toolbar">
            <div className="toolbar-segmented">
              {Object.entries(PACK_PRESETS).map(([key, cfg]) => (
                <button
                  key={key}
                  className={packPreset === key ? "active" : ""}
                  title={`Grid ${cfg.grid} | Angle ${cfg.angle}°`}
                  onClick={() => applyPackPreset(key, true)}
                >
                  {cfg.label}
                </button>
              ))}
            </div>
            <label className="toolbar-mini-input">
              Margin X
              <input
                type="number"
                min="0"
                step="1"
                value={packMarginX}
                onChange={(e) => setPackMarginX(Math.max(0, parseInt(e.target.value || "0", 10) || 0))}
              />
            </label>
            <label className="toolbar-mini-input">
              Margin Y
              <input
                type="number"
                min="0"
                step="1"
                value={packMarginY}
                onChange={(e) => setPackMarginY(Math.max(0, parseInt(e.target.value || "0", 10) || 0))}
              />
            </label>
            <button onClick={() => repackCurrentPacked(enableBleed)}>Repack</button>
          </div>
          ) : null}
          <div className={`preview region-stage ${rightPanelLoading ? "is-loading" : ""}`} ref={regionWrapRef} style={{ height: '100%'}}>
            <div className="preview-header">
              <div className="preview-title packed-title-row">
                <span>Packed (Konva)</span>
                {!sourceOnlyMode ? (
                  <>
                    <button
                      className={`icon-button ${packedEditMode === "move" ? "active" : ""}`}
                      onClick={() => setPackedEditMode((m) => (m === "move" ? "none" : "move"))}
                      title="Move zone"
                    >
                      Move
                    </button>
                    <button
                      className={`icon-button ${packedEditMode === "rotate" ? "active" : ""}`}
                      onClick={() => setPackedEditMode((m) => (m === "rotate" ? "none" : "rotate"))}
                      title="Rotate zone"
                    >
                      Rotate
                    </button>
                  </>
                ) : null}
              </div>
              {!sourceOnlyMode ? (
              <div className="preview-controls">
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showImages}
                    onChange={(e) => {
                      setShowImages(e.target.checked);
                    }}
                  />
                  Image
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showStroke}
                    onChange={(e) => {
                      setShowStroke(e.target.checked);
                    }}
                  />
                  Stroke
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={showLabels}
                    onChange={(e) => {
                      setShowLabels(e.target.checked);
                    }}
                  />
                  Label
                </label>
                <label className="checkbox">
                  <input
                    type="checkbox"
                    checked={enableBleed}
                    onChange={(e) => {
                      const checked = e.target.checked;
                      setEnableBleed(checked);
                      repackCurrentPacked(checked);
                    }}
                  />
                  Bleed
                </label>
                <label className="mini-input">
                  Font
                  <select
                    value={labelFontFamily}
                    onChange={(e) => setLabelFontFamily(e.target.value)}
                  >
                    <option value="Arial">Arial</option>
                    <option value="Helvetica">Helvetica</option>
                    <option value="Verdana">Verdana</option>
                    <option value="Tahoma">Tahoma</option>
                    <option value="Georgia">Georgia</option>
                    <option value="Times New Roman">Times New Roman</option>
                    <option value="Courier New">Courier New</option>
                  </select>
                </label>
                <label className="mini-input">
                  Size
                  <input
                    type="number"
                    min="4"
                    max="64"
                    value={toFinite(labelFontSize, 12)}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      setLabelFontSize(toFinite(v, 12));
                    }}
                  />
                </label>
                <button
                  className="icon-button"
                  title="Download"
                  onClick={() =>
                    downloadStage(
                      regionRef,
                      "packed-konva.svg",
                      scene?.canvas ? { w: scene.canvas.w, h: scene.canvas.h } : null
                    )
                  }
                >
                  {"\u2193"}
                </button>
              </div>
              ) : null}
            </div>
            {showRasterTemp ? (
              <div
                style={{
                  width: "100%",
                  height: "100%",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  overflow: "hidden",
                }}
              >
                <img
                  src={rasterTempSrc || "/out/tmp_raster_pack.png"}
                  alt="Raster temp"
                  style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                />
              </div>
            ) : packedSource ? (
              <Stage
                width={regionStageSize.w}
                height={regionStageSize.h}
                  draggable={packedEditMode === "none"}
                  scaleX={regionScale}
                  scaleY={regionScale}
                  x={regionPos.x}
                  y={regionPos.y}
                  onWheel={handleRegionWheel}
                  onDragMove={handleRegionDragMove}
                  onDragEnd={handleRegionDragMove}
                  onMouseMove={handlePackedStageMouseMove}
                  onMouseUp={handlePackedStageMouseUp}
                  onTouchMove={handlePackedStageMouseMove}
                  onTouchEnd={handlePackedStageMouseUp}
                ref={regionRef}
              ><Layer>{packedSource?.canvas ? (
                    <>
                      <Rect
                        x={0}
                        y={0}
                        width={packedSource.canvas.w}
                        height={packedSource.canvas.h}
                        stroke="#ffffff"
                        strokeWidth={2 / regionScale}
                        listening={false}
                      />
                    </>
                  ) : null}</Layer>{rightTab === "packed" ? (
                <Layer name="packed-image" visible={showImages}>{hasManualPackedEdits ? (
                    <>
                      {enableBleed &&
                        packedLiveBleedItems.map((it) => (
                        <Line
                          key={it.key}
                          points={it.points}
                          closed
                          fill={it.fill}
                          opacity={0.42}
                          listening={false}
                        />
                      ))}
                      {packedLiveFillItems.map((it) => (
                        <Line
                          key={it.key}
                          points={it.points}
                          closed
                          fill={it.fill}
                          strokeWidth={0}
                          listening={false}
                        />
                      ))}
                    </>
                  ) : (
                    <>
                      <Group
                        x={(packedSource?.canvas?.w || 0) / 2}
                        y={(packedSource?.canvas?.h || 0) / 2}
                        offsetX={(packedSource?.canvas?.w || 0) / 2}
                        offsetY={(packedSource?.canvas?.h || 0) / 2}
                      >
                        {packedFillPaths.map((p, idx) => (
                          <Path
                            key={`fill-path-${idx}`}
                            data={p.d}
                            fill={p.fill}
                            strokeWidth={0}
                            listening={false}
                          />
                        ))}
                        {enableBleed &&
                          packedBleedPaths.map((p, idx) => (
                          <Path
                            key={`bleed-path-${idx}`}
                            data={p.d}
                            fill={p.fill}
                            listening={false}
                          />
                        ))}
                      </Group>
                      <Group
                        x={(packedSource?.canvas?.w || 0) / 2 + (packedSource?.canvas?.w || 0) + 40}
                        y={(packedSource?.canvas?.h || 0) / 2}
                        offsetX={(packedSource?.canvas?.w || 0) / 2}
                        offsetY={(packedSource?.canvas?.h || 0) / 2}
                      >
                        {packedFillPaths2.map((p, idx) => (
                          <Path
                            key={`fill-path2-${idx}`}
                            data={p.d}
                            fill={p.fill}
                            strokeWidth={0}
                            listening={false}
                          />
                        ))}
                        {enableBleed &&
                          packedBleedPaths2.map((p, idx) => (
                          <Path
                            key={`bleed-path2-${idx}`}
                            data={p.d}
                            fill={p.fill}
                            listening={false}
                          />
                        ))}
                      </Group>
                    </>
                  )}</Layer>
                ) : null}{rightTab === "packed" && packedIndexItems.length ? (
                <Layer name="packed-label-snapped" visible={showLabels}>{packedIndexItems.map((lbl) => {
                    const size = Math.max(labelFontSize / regionScale, 6 / regionScale);
                    const metrics = measureText(lbl.label, size, labelFontFamily);
                    const isSelected = String(lbl.zid) === String(selectedZoneId);
                    return (
                      <Text
                        key={`psnap-${lbl.zid}`}
                        x={lbl.x}
                        y={lbl.y}
                        text={lbl.label}
                        fill={isSelected ? "#ff3b30" : "#ffffff"}
                        stroke="rgba(0,0,0,0.5)"
                        strokeWidth={1 / regionScale}
                        fontSize={size}
                        fontFamily={labelFontFamily}
                        align="center"
                        verticalAlign="middle"
                        offsetX={metrics.width / 2}
                        offsetY={metrics.height / 2}
                        listening
                        hitStrokeWidth={10 / regionScale}
                        onClick={() => handlePackedZoneSelect(lbl.zid)}
                        onTap={() => handlePackedZoneSelect(lbl.zid)}
                        onMouseDown={() => handlePackedZoneSelect(lbl.zid)}
                        onTouchStart={() => handlePackedZoneSelect(lbl.zid)}
                      />
                    );
                  })}</Layer>
                ) : null}{rightTab === "packed" ? (
                <Layer name="packed-overlay">{overlayItems.map((item) => {
                    if (!item?.img || item.zid == null) return null;
                    const packed = transformOverlayToPacked(item);
                    const bin =
                      packedSource?.placement_bin?.[item.zid] ??
                      packedSource?.placement_bin?.[parseInt(item.zid, 10)];
                    const page = bin === 1 ? 1 : 0;
                    const xOffset = page === 1 ? (packedSource?.canvas?.w || 0) + 40 : 0;
                    return (
                      <Image
                        key={`po-${item.id}`}
                        image={packed.img}
                        x={packed.x + xOffset}
                        y={packed.y}
                        width={packed.width}
                        height={packed.height}
                        offsetX={packed.width / 2}
                        offsetY={packed.height / 2}
                        scaleX={packed.scaleX}
                        scaleY={packed.scaleY}
                        rotation={packed.rotation}
                        listening={false}
                      />
                    );
                  })}</Layer>
                ) : null}{rightTab === "packed" ? (
                <Layer name="packed-stroke" visible={showStroke}>{packedZoneOutlineItems.map((item) => {
                    const isSelected = String(item.zid) === String(selectedZoneId);
                    return (
                      <Line
                        key={`pz-outline-${item.zid}-${item.idx}`}
                        points={toPoints(item.offsetPts)}
                        closed
                        stroke={isSelected ? "#ff3b30" : "#ffffff"}
                        strokeWidth={isSelected ? 3 : 1}
                        strokeScaleEnabled={false}
                        listening={false}
                      />
                    );
                  })}</Layer>
                ) : null}{rightTab === "packed" && packedLowAreaWarnings.length ? (
                <Layer name="packed-warning">{packedLowAreaWarnings.map((w) => {
                    const size = Math.max(28 / regionScale, 16 / regionScale);
                    const pad = Math.max(2 / regionScale, 1 / regionScale);
                    const triH = size * 0.8660254037844386;
                    const cx = w.maxx - pad - size / 2;
                    const cy = w.miny + pad + triH / 2;
                    return (
                      <React.Fragment key={`pw-${w.zid}`}>
                        <Line
                          points={[
                            cx, cy - triH / 2,
                            cx - size / 2, cy + triH / 2,
                            cx + size / 2, cy + triH / 2,
                          ]}
                          closed
                          fill="#ffd400"
                          stroke="#5a4a00"
                          strokeWidth={1 / regionScale}
                          strokeScaleEnabled={false}
                          listening={false}
                        />
                        <Text
                          x={cx}
                          y={cy + triH * 0.18}
                          text="!"
                          fill="#2a2100"
                          fontSize={Math.max(10 / regionScale, 6 / regionScale)}
                          fontFamily={labelFontFamily}
                          align="center"
                          verticalAlign="middle"
                          offsetX={0}
                          offsetY={0}
                          listening={false}
                        />
                      </React.Fragment>
                    );
                  })}</Layer>
                ) : null}{rightTab === "packed" ? (
                <Layer name="packed-hit">{packedZoneOutlineItems.map((item) => (
                    <Line
                      key={`pz-hit-${item.zid}-${item.idx}`}
                      points={toPoints(item.offsetPts)}
                      closed
                      fill="rgba(0,0,0,0)"
                      stroke="rgba(0,0,0,0)"
                      strokeWidth={8 / regionScale}
                      strokeScaleEnabled={false}
                      onClick={() => handlePackedZoneSelect(item.zid)}
                      onTap={() => handlePackedZoneSelect(item.zid)}
                      onMouseDown={(e) => beginPackedEdit(item.zid, e)}
                      onTouchStart={(e) => beginPackedEdit(item.zid, e)}
                    />
                  ))}</Layer>
                ) : null}{rightTab === "box" ? (
                <Layer name="packed-bbox">{showStroke
                    ? packedZoneOutlineItems.map((item) => {
                        const isSelected = String(item.zid) === String(selectedZoneId);
                        return (
                          <Line
                            key={`pb-outline-${item.zid}-${item.idx}`}
                            points={toPoints(item.offsetPts)}
                            closed
                            stroke={isSelected ? "#ff3b30" : "#ffffff"}
                            strokeWidth={isSelected ? 3 : 1}
                            strokeScaleEnabled={false}
                            listening={false}
                          />
                        );
                      })
                    : null}{packedEmptyCellsDerived.map((cell, idx) => (
                    <Circle
                      key={`pcell-${idx}`}
                      x={cell[0]}
                      y={cell[1]}
                      radius={2.2 / regionScale}
                      stroke="#00c2ff"
                      strokeWidth={1 / regionScale}
                      fill="rgba(0,194,255,0.12)"
                      listening={false}
                    />
                  ))}{packedBoxData.map((box) => {
                    const w = Math.max(0, box.maxx - box.minx);
                    const h = Math.max(0, box.maxy - box.miny);
                    const size = Math.max(10 / regionScale, 6 / regionScale);
                    const metrics = measureText(box.label, size, labelFontFamily);
                    return (
                      <React.Fragment key={`pb-${box.zid}`}>
                        {showStroke ? (
                          <Rect
                            x={box.minx + box.xOffset}
                            y={box.miny}
                            width={w}
                            height={h}
                            stroke="#00ff7f"
                            strokeWidth={1 / regionScale}
                            listening={false}
                          />
                        ) : null}
                        {showLabels ? (
                          <Text
                            x={box.minx + box.xOffset + 2 / regionScale}
                            y={box.miny + 2 / regionScale}
                            text={`${box.label}`}
                            fill="#00ff7f"
                            fontSize={size}
                            fontFamily={labelFontFamily}
                            align="left"
                            verticalAlign="top"
                            offsetX={0}
                            offsetY={0}
                            listening={false}
                          />
                        ) : null}
                      </React.Fragment>
                    );
                  })}</Layer>
                ) : null}</Stage>
            ) : null}
            {renderLoadingOverlay("right")}
            {enableBleed && packedBleedError ? <div className="error">{packedBleedError}</div> : null}
            {enableBleed && packedBleedError2 ? <div className="error">{packedBleedError2}</div> : null}
          </div>
        </div>
      </div>
      {exportPdfLoading || exportPdfInfo ? (
        <div className="modal-backdrop">
          <div className="modal">
            <div className="modal-title">
              {exportPdfLoading
                ? `Creating PDF... ${Math.max(0, 60 - exportPdfTiming.elapsedMs / 1000).toFixed(1)}s`
                : "Successful created PDF !"}
            </div>
            {!exportPdfLoading && exportPdfInfo ? (
              <div className="modal-actions">
                <button
                  className="btn ghost"
                  onClick={() => setExportPdfInfo(null)}
                >
                  Cancel
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    window.location = `/api/download_pdf?name=${encodeURIComponent(
                      exportPdfInfo.name
                    )}`;
                    setExportPdfInfo(null);
                  }}
                >
                  Download PDF
                </button>
                {exportHtmlInfo.map((name) => (
                  <button
                    key={name}
                    className="btn"
                    onClick={() => {
                      window.location = `/api/download_html?name=${encodeURIComponent(name)}`;
                    }}
                  >
                    Download HTML
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
      {showSim ? (
        <div className="modal-backdrop">
          <div className="modal sim-modal">
            <button className="modal-close" onClick={() => setShowSim(false)}>
              X
            </button>
            <div className="modal-title">Simulate</div>
            <div className="sim-status">
              {simActiveLabel ? `Moving index: ${simActiveLabel}` : "Moving index: -"}
            </div>
            <div className="sim-body" ref={simWrapRef}>
              {simStage}
            </div>
            <div className="sim-controls">
              <button className="icon-button" onClick={handleSimPlayToggle}>
                {simPlaying ? (
                  <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                    <rect x="4" y="3" width="4" height="14" fill="currentColor" />
                    <rect x="12" y="3" width="4" height="14" fill="currentColor" />
                  </svg>
                ) : (
                  <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
                    <polygon points="6,4 16,10 6,16" fill="currentColor" />
                  </svg>
                )}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.001"
                value={simProgress}
                onChange={(e) => setSimProgress(parseFloat(e.target.value))}
              />
              <button
                className="btn"
                onClick={handleSimVideoDownload}
                disabled={simVideoLoading}
              >
                {simVideoLoading ? "Creating..." : "Download GIF"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}


