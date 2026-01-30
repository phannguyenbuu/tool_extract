import React, { useMemo, useRef, useState, useEffect } from "react";
import { Stage, Layer, Line, Text, Circle, Rect } from "react-konva";

const toPoints = (pts) => pts.flatMap((p) => [p[0], p[1]]);

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

export default function App() {
  const [snap, setSnap] = useState(1);
  const [scene, setScene] = useState(null);
  const [error, setError] = useState("");
  const [labels, setLabels] = useState([]);
  const [rawSegments, setRawSegments] = useState([]);
  const [borderSegments, setBorderSegments] = useState([]);
  const [nodes, setNodes] = useState([]);
  const [segs, setSegs] = useState([]);
  const [svgImage, setSvgImage] = useState(null);
  const [svgFallback, setSvgFallback] = useState([]);
  const [svgSize, setSvgSize] = useState({ w: 1000, h: 1000 });
  const stageRef = useRef(null);
  const leftRef = useRef(null);
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
  const [autoFit, setAutoFit] = useState(true);
  const [showImages, setShowImages] = useState(false);
  const [zoneLoading, setZoneLoading] = useState(false);
  const [packedLoading, setPackedLoading] = useState(false);
  const [zoneSrc, setZoneSrc] = useState("/out/zone_outline.svg");
  const [packedSrc, setPackedSrc] = useState("/out/packed.svg");
  const [zoneRetry, setZoneRetry] = useState(false);
  const [edgeMode, setEdgeMode] = useState(false);
  const [edgeCandidate, setEdgeCandidate] = useState(null);
  const [sceneLoading, setSceneLoading] = useState(true);
  const [packPadding, setPackPadding] = useState(4);
  const [packMarginX, setPackMarginX] = useState(30);
  const [packMarginY, setPackMarginY] = useState(30);
  const [packBleed, setPackBleed] = useState(10);
  const [packGrid, setPackGrid] = useState(5);
  const [packAngle, setPackAngle] = useState(5);
  const [packMode, setPackMode] = useState("fast");
  const [autoPack, setAutoPack] = useState(false);

  useEffect(() => {
    loadScene();
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

  const loadScene = async (fit = true) => {
    try {
      setError("");
      setAutoFit(fit);
      setSceneLoading(true);
      const svgRes = await fetch("/out/convoi.svg");
      if (!svgRes.ok) throw new Error(`svg fetch failed: ${svgRes.status}`);
      const svgText = await svgRes.text();
      const parsedSize = parseSvgSize(svgText);
      setSvgSize(parsedSize);
      const parsed = buildSegmentsFromSvg(svgText);
      const segments = parsed.segments;
      const borders = parsed.borderSegments;
      setSvgFallback(segments);
      setBorderSegments(borders);
      // no background rendering; keep only geometry
      setRawSegments(segments);
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
      const snapped = snapNodes(splitSegments, snap);
      setNodes(snapped.nodes);
      setSegs(snapped.segs);

      const res = await fetch(
        `/api/scene?snap=${snap}&pack_padding=${packPadding}&pack_margin_x=${packMarginX}&pack_margin_y=${packMarginY}&pack_bleed=${packBleed}&pack_grid=${packGrid}&pack_angle=${packAngle}&pack_mode=${packMode}`
      );
      if (!res.ok) {
        throw new Error(`scene fetch failed: ${res.status}`);
      }
      const data = await res.json();
      setScene(data);
      const initLabels = Object.values(data.zone_labels || {}).map((v) => ({
        id: `z-${v.label}`,
        x: v.x,
        y: v.y,
        label: `${v.label}`,
      }));
      setLabels(initLabels);
      if (fit) {
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
        const packedBounds = calcBounds(
          data.packed_zone_polys && data.packed_zone_polys.length
            ? data.packed_zone_polys
            : data.packed_polys && data.packed_polys.length
            ? data.packed_polys
            : data.regions || []
        );
        fitRegionToView(packedBounds);
        const regionBounds = calcBounds(data.regions || []);
        fitRegion2ToView(regionBounds);
        const zoneBounds = calcBoundsFromLines(data.zone_boundaries);
        fitZoneToView(zoneBounds);
      }
      setSceneLoading(false);
    } catch (err) {
      setError(err.message || String(err));
      setSceneLoading(false);
    }
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

  const handleRegion2Wheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = region2Ref.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setRegion2Scale(newScale);
    setRegion2Pos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const handleZoneWheel = (e) => {
    e.evt.preventDefault();
    const scaleBy = 1.05;
    const stage = zoneRef.current;
    const oldScale = stage.scaleX();
    const pointer = stage.getPointerPosition();
    const mousePointTo = {
      x: (pointer.x - stage.x()) / oldScale,
      y: (pointer.y - stage.y()) / oldScale,
    };
    const direction = e.evt.deltaY > 0 ? 1 : -1;
    const newScale = direction > 0 ? oldScale / scaleBy : oldScale * scaleBy;
    setZoneScale(newScale);
    setZonePos({
      x: pointer.x - mousePointTo.x * newScale,
      y: pointer.y - mousePointTo.y * newScale,
    });
  };

  const saveState = async () => {
    if (!scene) return;
    await fetch("/api/state", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        canvas: scene.canvas,
        regions: scene.regions,
        zone_boundaries: scene.zone_boundaries,
        svg_nodes: nodes,
        svg_segments: segs,
        labels,
        snap,
      }),
    });
  };

  const renderPNGs = async () => {
    setZoneLoading(true);
    setPackedLoading(true);
    setZoneRetry(false);
    await fetch("/api/render", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        snap,
        pack_padding: packPadding,
        pack_margin_x: packMarginX,
        pack_margin_y: packMarginY,
        pack_bleed: packBleed,
        pack_grid: packGrid,
        pack_angle: packAngle,
        pack_mode: packMode,
      }),
    });
    const ts = Date.now();
    setZoneSrc(`${showImages ? "/out/zone.svg" : "/out/zone_outline.svg"}?t=${ts}`);
    setPackedSrc(`${showImages ? "/out/packed.svg" : "/out/packed_outline.png"}?t=${ts}`);
  };

  useEffect(() => {
    if (!autoPack) return;
    const id = setTimeout(() => {
      renderPNGs().then(() => loadScene(false));
    }, 500);
    return () => clearTimeout(id);
  }, [packPadding, packMarginX, packMarginY, packBleed, packGrid, packAngle, packMode, autoPack]);

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
          strokeWidth={(1 / scale) * 2}
          strokeScaleEnabled={false}
        />
      );
    });
  }, [segs, nodes]);

  const borderLayer = useMemo(() => {
    if (!borderSegments.length) return null;
    return borderSegments.map((seg, idx) => (
      <Line
        key={`b-${idx}`}
        points={toPoints(seg)}
        stroke="#f5f6ff"
        strokeWidth={(1 / scale) * 2}
        strokeScaleEnabled={false}
      />
    ));
  }, [borderSegments]);

  useEffect(() => {
    if (autoFit && (scene?.packed_polys?.length || scene?.regions?.length)) {
      const packedBounds = calcBounds(
        scene.packed_zone_polys && scene.packed_zone_polys.length
          ? scene.packed_zone_polys
          : scene.packed_polys && scene.packed_polys.length
          ? scene.packed_polys
          : scene.regions
      );
      fitRegionToView(packedBounds);
      fitRegion2ToView(calcBounds(scene.regions || []));
      fitZoneToView(calcBoundsFromLines(scene.zone_boundaries));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, regionStageSize, region2StageSize, zoneStageSize, autoFit]);

  return (
    <div className="app">
      <div className="panel toolbar">
        <label>Snap</label>
        <input
          value={snap}
          onChange={(e) => setSnap(parseFloat(e.target.value || "0"))}
          type="number"
          step="0.1"
        />
        <button onClick={loadScene}>Load</button>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={showImages}
              onChange={(e) => {
                setShowImages(e.target.checked);
                setZoneLoading(true);
                setPackedLoading(true);
                setZoneRetry(false);
                const ts = Date.now();
                setZoneSrc(`${e.target.checked ? "/out/zone.svg" : "/out/zone_outline.svg"}?t=${ts}`);
                setPackedSrc(`${e.target.checked ? "/out/packed.svg" : "/out/packed_outline.png"}?t=${ts}`);
              }}
          />
          Image
        </label>
        <button onClick={renderPNGs}>Render PNGs</button>
        <button onClick={saveState}>Save JSON+SVG</button>
        <button
          className={edgeMode ? "active" : ""}
          onClick={() => {
            setEdgeMode((v) => !v);
            setEdgeCandidate(null);
          }}
        >
          Create Edge
        </button>
        <label>Pad</label>
        <input type="number" step="0.5" value={packPadding} onChange={(e) => setPackPadding(parseFloat(e.target.value || "0"))} />
        <label>MX</label>
        <input type="number" step="1" value={packMarginX} onChange={(e) => setPackMarginX(parseFloat(e.target.value || "0"))} />
        <label>MY</label>
        <input type="number" step="1" value={packMarginY} onChange={(e) => setPackMarginY(parseFloat(e.target.value || "0"))} />
        <label>Bleed</label>
        <input type="number" step="1" value={packBleed} onChange={(e) => setPackBleed(parseFloat(e.target.value || "0"))} />
        <label>Grid</label>
        <input type="number" step="1" value={packGrid} onChange={(e) => setPackGrid(parseFloat(e.target.value || "0"))} />
        <label>Angle</label>
        <input type="number" step="1" value={packAngle} onChange={(e) => setPackAngle(parseFloat(e.target.value || "0"))} />
        <label>Mode</label>
        <select value={packMode} onChange={(e) => setPackMode(e.target.value)}>
          <option value="fast">fast</option>
          <option value="full">full</option>
        </select>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={autoPack}
            onChange={(e) => setAutoPack(e.target.checked)}
          />
          Auto pack
        </label>
        <button onClick={() => renderPNGs().then(() => loadScene(false))}>Apply Pack</button>
        <div className="toolbar-spacer" />
        {error ? <div className="error">{error}</div> : null}
      </div>

      <div className="content">
        <div className={`left ${sceneLoading ? "is-loading" : ""}`} ref={leftRef}>
          <Stage
            width={stageSize.w}
            height={stageSize.h}
            draggable
            scaleX={scale}
            scaleY={scale}
            x={pos.x}
            y={pos.y}
            onWheel={handleWheel}
            onMouseMove={(e) => {
              if (!edgeMode) return;
              const stage = stageRef.current;
              const pointer = stage.getPointerPosition();
              if (!pointer) return;
              const world = {
                x: (pointer.x - pos.x) / scale,
                y: (pointer.y - pos.y) / scale,
              };
              const cand = findEdgeCandidate(world);
              setEdgeCandidate(cand);
            }}
            onMouseLeave={() => {
              if (edgeMode) setEdgeCandidate(null);
            }}
            onMouseDown={() => {
              if (!edgeMode || !edgeCandidate) return;
              const key = edgeKey(edgeCandidate.a, edgeCandidate.b);
              const segSet = new Set(segs.map(([a, b]) => edgeKey(a, b)));
              if (segSet.has(key)) return;
              const nextSegs = [...segs, [edgeCandidate.a, edgeCandidate.b]];
              setSegs(nextSegs);
              fetch("/api/save_svg", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ nodes, segs: nextSegs }),
              }).then(() => loadScene(false));
            }}
            ref={stageRef}
          >
        <Layer>{nodeLayer}</Layer>
        <Layer>{borderLayer}</Layer>
        {edgeCandidate ? (
          <Layer>
            <Line
              points={[
                nodes[edgeCandidate.a].x,
                nodes[edgeCandidate.a].y,
                nodes[edgeCandidate.b].x,
                nodes[edgeCandidate.b].y,
              ]}
              stroke="#cfd6ff"
              opacity={0.4}
              strokeWidth={(1 / scale) * 2}
              strokeScaleEnabled={false}
            />
          </Layer>
        ) : null}
        <Layer>
          {nodes.map((n) => (
            <Circle
              key={`n-${n.id}`}
              x={n.x}
              y={n.y}
              radius={3 / scale}
              fill="red"
              strokeScaleEnabled={false}
              draggable={!edgeMode}
              onDragMove={(e) => {
                const next = nodes.map((p) =>
                  p.id === n.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
                );
                setNodes(next);
              }}
              onDragEnd={(e) => {
                const next = nodes.map((p) =>
                  p.id === n.id ? { ...p, x: e.target.x(), y: e.target.y() } : p
                );
                const merged = mergeNodesIfClose(next, segs, n.id, snap);
                setNodes(merged.nodes);
                setSegs(merged.segs);
                fetch("/api/save_svg", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ nodes: merged.nodes, segs: merged.segs }),
                }).then(() => loadScene(false));
              }}
            />
          ))}
        </Layer>
          </Stage>
          {sceneLoading ? <div className="loading-overlay">Loading…</div> : null}
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
                ? ` empty=${(scene.debug.zones_empty || []).length} hull=${(scene.debug.zones_convex_hull || []).length}`
                : " n/a"}
            </div>
          </div>
        </div>
        <div className="right">
          <div className={`preview tall region-stage ${sceneLoading ? "is-loading" : ""}`} ref={regionWrapRef}>
            <div className="preview-title">Packed (Konva)</div>
            {scene ? (
              <Stage
                width={regionStageSize.w}
                height={regionStageSize.h}
                draggable
                scaleX={regionScale}
                scaleY={regionScale}
                x={regionPos.x}
                y={regionPos.y}
                onWheel={handleRegionWheel}
                ref={regionRef}
              >
                <Layer>
                  {scene?.canvas ? (
                    <Rect
                      x={0}
                      y={0}
                      width={scene.canvas.w}
                      height={scene.canvas.h}
                      stroke="#f5f6ff"
                      strokeWidth={1 / regionScale}
                      strokeScaleEnabled={false}
                    />
                  ) : null}
                </Layer>
                <Layer>
                  {showImages && scene.packed_polys && scene.packed_colors
                    ? scene.packed_polys.map((poly, idx) => (
                        <Line
                          key={`pf-${idx}`}
                          points={toPoints(poly)}
                          closed
                          fill={scene.packed_colors[idx]}
                          strokeScaleEnabled={false}
                        />
                      ))
                    : null}
                </Layer>
                <Layer>
                  {(scene.packed_zone_polys || []).map((poly, idx) => (
                    <Line
                      key={`pz-${idx}`}
                      points={toPoints(poly)}
                      closed
                      stroke="#f5f6ff"
                      strokeWidth={1 / regionScale}
                      strokeScaleEnabled={false}
                    />
                  ))}
                </Layer>
              </Stage>
              ) : null}
            {sceneLoading ? <div className="loading-overlay">Loading…</div> : null}
          </div>
          <div className="preview-row">
            <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={region2WrapRef}>
              <div className="preview-title">Region (Konva)</div>
              {scene ? (
                <Stage
                  width={region2StageSize.w}
                  height={region2StageSize.h}
                  draggable
                  scaleX={region2Scale}
                  scaleY={region2Scale}
                  x={region2Pos.x}
                  y={region2Pos.y}
                  onWheel={handleRegion2Wheel}
                  ref={region2Ref}
                >
                  <Layer>
                    {scene.regions.map((poly, idx) => (
                      <Line
                        key={`r2-${idx}`}
                        points={toPoints(poly)}
                        closed
                        stroke="#f5f6ff"
                        fill="#bbb"
                        strokeWidth={1 / region2Scale}
                        strokeScaleEnabled={false}
                      />
                    ))}
                  </Layer>
                </Stage>
              ) : null}
              {sceneLoading ? <div className="loading-overlay">Loading…</div> : null}
            </div>
            <div className={`preview half ${sceneLoading ? "is-loading" : ""}`} ref={zoneWrapRef}>
              <div className="preview-title">Zone (Konva)</div>
              {scene ? (
                <Stage
                  width={zoneStageSize.w}
                  height={zoneStageSize.h}
                  draggable
                  scaleX={zoneScale}
                  scaleY={zoneScale}
                  x={zonePos.x}
                  y={zonePos.y}
                  onWheel={handleZoneWheel}
                  ref={zoneRef}
                >
                  <Layer>
                    {showImages && scene.region_colors
                      ? scene.regions.map((poly, idx) => (
                          <Line
                            key={`zf-${idx}`}
                            points={toPoints(poly)}
                            closed
                            fill={scene.region_colors[idx]}
                            strokeScaleEnabled={false}
                          />
                        ))
                      : null}
                  </Layer>
                  <Layer>
                    {Object.entries(scene.zone_boundaries || {}).flatMap(([zid, paths]) =>
                      paths.map((p, i) => (
                        <Line
                          key={`zb2-${zid}-${i}`}
                          points={toPoints(p)}
                        stroke="#f5f6ff"
                          strokeWidth={1 / zoneScale}
                          strokeScaleEnabled={false}
                        />
                      ))
                    )}
                  </Layer>
                </Stage>
              ) : null}
              {sceneLoading ? <div className="loading-overlay">Loading…</div> : null}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
