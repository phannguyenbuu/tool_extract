import { SVGShapeData, PackedItem } from '../types';

interface ShapeMask {
  width: number;
  height: number;
  data: Uint8Array; // 1 = occupied, 0 = empty
}

export class RasterPacker {
  width: number;
  height: number;
  grid: Uint8Array;
  maskCache: Record<string, ShapeMask> = {};

  constructor(width: number, height: number) {
    this.width = Math.ceil(width);
    this.height = Math.ceil(height);
    this.grid = new Uint8Array(this.width * this.height);
  }

  private getMaskKey(shapeId: string, rotation: number, scale: number, gap: number): string {
    return `${shapeId}-${rotation}-${scale}-${gap.toFixed(2)}`;
  }

  async precomputeMask(shape: SVGShapeData, rotation: number, gapMM: number = 0) {
    const key = this.getMaskKey(shape.id, rotation, shape.scale, gapMM);
    if (this.maskCache[key]) return;

    // Use tight bbox dimensions
    const vbW = shape.bbox.w;
    const vbH = shape.bbox.h;
    const vbX = shape.bbox.x;
    const vbY = shape.bbox.y;
    
    // Physical dimensions with User Scale
    const userScale = (shape.scale || 100) / 100;
    const mmW = shape.widthMM * userScale;
    const mmH = shape.heightMM * userScale;
    
    // Scale factor from SVG BBox units to Pixels (mm)
    // The raster grid has 1 pixel = 1 mm resolution
    const scaleX = mmW / vbW;
    const scaleY = mmH / vbH;

    // Calculate rotated bounding box size
    const rad = (rotation * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);

    const corners = [
        { x: -mmW/2, y: -mmH/2 },
        { x: mmW/2, y: -mmH/2 },
        { x: mmW/2, y: mmH/2 },
        { x: -mmW/2, y: mmH/2 }
    ];

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    
    corners.forEach(c => {
        const rx = c.x * cos - c.y * sin;
        const ry = c.x * sin + c.y * cos;
        if (rx < minX) minX = rx;
        if (rx > maxX) maxX = rx;
        if (ry < minY) minY = ry;
        if (ry > maxY) maxY = ry;
    });

    // Padding Logic: gapMM is the required empty space around the shape in MM.
    // The mask needs to represent the "Occupied Area" including this buffer.
    // We expand the canvas to fit the shape + buffer.
    const gapPx = Math.ceil(gapMM); 

    const canvasW = Math.ceil(maxX - minX) + (gapPx * 2);
    const canvasH = Math.ceil(maxY - minY) + (gapPx * 2);

    const canvas = document.createElement('canvas');
    canvas.width = canvasW;
    canvas.height = canvasH;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    
    if (!ctx) throw new Error("Could not create canvas context");

    ctx.clearRect(0,0, canvasW, canvasH);
    
    // Configure context for masking
    // Color > 10 in alpha/red will be considered occupied
    ctx.fillStyle = '#000000';
    ctx.strokeStyle = '#000000';
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    
    ctx.save();
    
    // 1. Move to center of canvas
    ctx.translate(canvasW / 2, canvasH / 2);
    
    // 2. Rotate
    ctx.rotate(rad);
    
    // 3. Center the shape physically before scaling/unshifting
    ctx.translate(-mmW / 2, -mmH / 2);
    
    // 4. Scale from SVG units to MM
    ctx.scale(scaleX, scaleY);
    
    // 5. Shift back by the SVG ViewBox origin
    ctx.translate(-vbX, -vbY);
    
    // Correctly calculate lineWidth in the SCALED coordinate system.
    // We want the physical stroke width to be (gapMM * 2).
    // EffectiveWidth = LineWidth * Scale.
    // => LineWidth = (gapMM * 2) / Scale.
    // We use scaleX (assuming uniform scale for stroke)
    if (gapMM > 0) {
        ctx.lineWidth = (gapMM * 2) / scaleX;
    }
    
    const p = new Path2D(shape.path);
    
    if (gapMM > 0) {
        ctx.stroke(p);
    }
    ctx.fill(p);
    
    ctx.restore();

    const imgData = ctx.getImageData(0, 0, canvasW, canvasH);
    const data = imgData.data; 
    const maskData = new Uint8Array(canvasW * canvasH);

    // Threshold logic: Any non-transparent pixel is part of the mask
    for (let i = 0; i < maskData.length; i++) {
        // Check alpha or color channel
        if (data[i * 4 + 3] > 10) { 
            maskData[i] = 1;
        }
    }

    this.maskCache[key] = {
        width: canvasW,
        height: canvasH,
        data: maskData
    };
  }

  findPosition(shapeId: string, rotation: number, scale: number, gap: number): { x: number, y: number } | null {
    const mask = this.maskCache[this.getMaskKey(shapeId, rotation, scale, gap)];
    if (!mask) return null;

    const limitY = this.height - mask.height;
    const limitX = this.width - mask.width;

    for (let y = 0; y <= limitY; y++) {
      for (let x = 0; x <= limitX; x++) {
        // Optimization: Quick check if top-left is empty (not perfect but speeds up)
        if (this.grid[y * this.width + x] === 1) { x++; continue; } 
        
        if (this.fits(x, y, mask)) {
            return { x, y };
        }
      }
    }
    return null;
  }

  private fits(gx: number, gy: number, mask: ShapeMask): boolean {
    const grid = this.grid;
    const gridWidth = this.width;
    const maskData = mask.data;
    const maskW = mask.width;
    const maskH = mask.height;

    // Check intersection
    for (let my = 0; my < maskH; my++) {
        const gridRowStart = (gy + my) * gridWidth + gx;
        const maskRowStart = my * maskW;
        
        for (let mx = 0; mx < maskW; mx++) {
            if (maskData[maskRowStart + mx] === 1) {
                if (grid[gridRowStart + mx] === 1) {
                    return false;
                }
            }
        }
    }
    return true;
  }

  place(shapeId: string, rotation: number, scale: number, gap: number, x: number, y: number) {
    const mask = this.maskCache[this.getMaskKey(shapeId, rotation, scale, gap)];
    if (!mask) return;

    for (let my = 0; my < mask.height; my++) {
        const rowOffsetMask = my * mask.width;
        const rowOffsetGrid = (y + my) * this.width + x;
        for (let mx = 0; mx < mask.width; mx++) {
            if (mask.data[rowOffsetMask + mx] === 1) {
                this.grid[rowOffsetGrid + mx] = 1;
            }
        }
    }
  }

  insert(shape: SVGShapeData, gap: number = 0): PackedItem | null {
    const ROTATIONS = [0, 90, 180, 270, 45, 135, 225, 315]; 
    const scale = shape.scale || 100;

    let bestMove: { x: number, y: number, rotation: number, score: number } | null = null;

    for (const rot of ROTATIONS) {
        const pos = this.findPosition(shape.id, rot, scale, gap);
        if (pos) {
            // Heuristic: top-left preference (minimize y*width + x)
            const score = pos.y * this.width + pos.x;
            if (!bestMove || score < bestMove.score) {
                bestMove = { x: pos.x, y: pos.y, rotation: rot, score };
                // Greedy early exit if perfect fit at 0,0 (optional)
                if (score === 0) break;
            }
        }
    }

    if (bestMove) {
        this.place(shape.id, bestMove.rotation, scale, gap, bestMove.x, bestMove.y);
        const mask = this.maskCache[this.getMaskKey(shape.id, bestMove.rotation, scale, gap)];
        
        return {
            shapeId: shape.id,
            // The position returned is the Top-Left of the MASK (which includes padding)
            x: bestMove.x, 
            y: bestMove.y,
            width: mask.width,
            height: mask.height,
            rotation: bestMove.rotation
        };
    }

    return null;
  }
}