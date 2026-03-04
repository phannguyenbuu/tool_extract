import { SVGShapeData } from '../types';

/**
 * Extracts numeric values from a viewBox string
 */
function parseViewBox(viewBox: string) {
  const parts = viewBox.split(/[\s,]+/).map(Number);
  return { x: parts[0] || 0, y: parts[1] || 0, width: parts[2] || 0, height: parts[3] || 0 };
}

/**
 * Converts basic SVG shapes to Path data (d attribute)
 */
function convertToPath(element: Element): string | null {
  const tag = element.tagName.toLowerCase();
  
  if (tag === 'path') {
    return element.getAttribute('d');
  }

  // Convert Rect to Path
  if (tag === 'rect') {
    const x = Number(element.getAttribute('x')) || 0;
    const y = Number(element.getAttribute('y')) || 0;
    const w = Number(element.getAttribute('width')) || 0;
    const h = Number(element.getAttribute('height')) || 0;
    return `M${x},${y} h${w} v${h} h-${w} Z`;
  }

  // Convert Circle to Path
  if (tag === 'circle') {
    const cx = Number(element.getAttribute('cx')) || 0;
    const cy = Number(element.getAttribute('cy')) || 0;
    const r = Number(element.getAttribute('r')) || 0;
    return `M${cx - r},${cy} a${r},${r} 0 1,0 ${r * 2},0 a${r},${r} 0 1,0 -${r * 2},0`;
  }

  // Convert Ellipse to Path
  if (tag === 'ellipse') {
      const cx = Number(element.getAttribute('cx')) || 0;
      const cy = Number(element.getAttribute('cy')) || 0;
      const rx = Number(element.getAttribute('rx')) || 0;
      const ry = Number(element.getAttribute('ry')) || 0;
      return `M${cx - rx},${cy} a${rx},${ry} 0 1,0 ${rx * 2},0 a${rx},${ry} 0 1,0 -${rx * 2},0`;
  }

  // Convert Polygon/Polyline to Path
  if (tag === 'polygon' || tag === 'polyline') {
    const points = element.getAttribute('points');
    if (!points) return null;
    const coords = points.trim().split(/[\s,]+/);
    let d = `M${coords[0]},${coords[1]}`;
    for (let i = 2; i < coords.length; i += 2) {
      d += ` L${coords[i]},${coords[i + 1]}`;
    }
    if (tag === 'polygon') d += ' Z';
    return d;
  }
  
  // Convert Image to Rect Path (for collision detection ONLY if no vectors exist)
  if (tag === 'image') {
      const x = Number(element.getAttribute('x')) || 0;
      const y = Number(element.getAttribute('y')) || 0;
      const w = Number(element.getAttribute('width')) || 0;
      const h = Number(element.getAttribute('height')) || 0;
      if (w === 0 || h === 0) return null;
      return `M${x},${y} h${w} v${h} h-${w} Z`;
  }

  return null;
}

export const parseSVGFile = async (file: File): Promise<SVGShapeData | null> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const text = e.target?.result as string;
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'image/svg+xml');
        const svg = doc.querySelector('svg');

        if (!svg) {
          reject(new Error('Invalid SVG file'));
          return;
        }

        // 1. Get Initial ViewBox and Dimensions from file attributes
        let viewBoxStr = svg.getAttribute('viewBox');
        let widthStr = svg.getAttribute('width');
        let heightStr = svg.getAttribute('height');

        let originalVbW = 0;
        let originalVbH = 0;

        // Parse explicit attributes first to establish scale
        if (viewBoxStr) {
          const vb = parseViewBox(viewBoxStr);
          originalVbW = vb.width;
          originalVbH = vb.height;
        } else if (widthStr && heightStr) {
          originalVbW = parseFloat(widthStr);
          originalVbH = parseFloat(heightStr);
          viewBoxStr = `0 0 ${originalVbW} ${originalVbH}`;
        } else {
           originalVbW = 100; originalVbH = 100; viewBoxStr = "0 0 100 100";
        }

        // Unit parser
        const parseUnit = (str: string | null, ref: number) => {
            if (!str) return ref; 
            if (str.endsWith('mm')) return parseFloat(str);
            if (str.endsWith('cm')) return parseFloat(str) * 10;
            if (str.endsWith('in')) return parseFloat(str) * 25.4;
            if (str.endsWith('px')) return parseFloat(str) * 0.264583;
            return parseFloat(str); 
        };

        const originalWidthMM = parseUnit(widthStr, originalVbW);
        const originalHeightMM = parseUnit(heightStr, originalVbH);

        // 2. Measure actual Bounding Box (Trimming Whitespace)
        const sandbox = document.createElement('div');
        sandbox.style.position = 'absolute';
        sandbox.style.visibility = 'hidden';
        sandbox.style.pointerEvents = 'none';
        sandbox.style.top = '-9999px';
        sandbox.style.left = '-9999px';
        
        const tempSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        tempSvg.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
        tempSvg.innerHTML = svg.innerHTML;
        
        // Prioritize vector shapes for cutting contour
        const vectorSelectors = 'path, rect, circle, polygon, polyline, ellipse';
        const hasVectors = tempSvg.querySelector(vectorSelectors) !== null;

        if (hasVectors) {
            // Remove images from measurement sandbox so we only measure the cut lines
            // This ensures bbox is tight to vectors, ignoring large image backgrounds
            const images = tempSvg.querySelectorAll('image');
            images.forEach(img => img.parentNode?.removeChild(img));
        }

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        while (tempSvg.firstChild) {
            g.appendChild(tempSvg.firstChild);
        }
        tempSvg.appendChild(g);
        sandbox.appendChild(tempSvg);
        document.body.appendChild(sandbox);

        // Calculate Scale Factor (Physical MM per ViewBox Unit)
        const scaleX = originalWidthMM / (originalVbW || 1);
        const scaleY = originalHeightMM / (originalVbH || 1);

        let bbox = { x: 0, y: 0, width: originalVbW, height: originalVbH };
        let finalWidthMM = originalWidthMM;
        let finalHeightMM = originalHeightMM;

        try {
            const tempBBox = g.getBBox();
            if (tempBBox.width > 0 && tempBBox.height > 0) {
                // We use the tight bbox for dimensions
                bbox = { x: tempBBox.x, y: tempBBox.y, width: tempBBox.width, height: tempBBox.height };
                
                // Update Physical Dimensions based on BBox size * original scale
                const margin = 0; // No extra margin on physics, handled by padding in packer
                finalWidthMM = bbox.width * scaleX;
                finalHeightMM = bbox.height * scaleY;
            }
        } catch (err) {
            console.warn("Could not calculate BBox", err);
        }
        
        document.body.removeChild(sandbox);

        // 3. Extract collision path
        const shapeSelector = hasVectors ? vectorSelectors : 'path, rect, circle, polygon, polyline, image';
        const shapes = doc.querySelectorAll(shapeSelector);
        let combinedPath = '';
        
        for (let i = 0; i < shapes.length; i++) {
            const d = convertToPath(shapes[i]);
            if (d) {
                combinedPath += ` ${d}`;
            }
        }

        if (!combinedPath) {
             combinedPath = `M0,0 h${originalVbW} v${originalVbH} h-${originalVbW} Z`;
        }

        const svgContent = svg.innerHTML;

        // Generate a random color
        const colors = ["#EF476F", "#FFD166", "#06D6A0", "#118AB2", "#073B4C", "#9D4EDD", "#FF9F1C"];
        const randomColor = colors[Math.floor(Math.random() * colors.length)];

        resolve({
          id: `shape_${Date.now()}_${Math.random().toString(36).substr(2, 5)}`,
          name: file.name.replace('.svg', ''),
          viewBox: viewBoxStr, // Keep ORIGINAL ViewBox for rendering
          bbox: { x: bbox.x, y: bbox.y, w: bbox.width, h: bbox.height }, // Keep TIGHT BBox for packing
          widthMM: finalWidthMM,
          heightMM: finalHeightMM,
          path: combinedPath.trim(),
          svgContent: svgContent, 
          color: randomColor,
          strokeWidth: 1, 
          quantity: 1, 
          ratio: 0, 
          scale: 100 
        });

      } catch (err) {
        reject(err);
      }
    };
    reader.readAsText(file);
  });
};