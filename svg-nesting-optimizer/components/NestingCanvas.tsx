import React, { useMemo, useRef, useEffect, useState } from 'react';
import { SVGShapeData, PackedItem } from '../types';
import { ZoomIn, ZoomOut, Maximize } from 'lucide-react';

interface NestingCanvasProps {
  shapes: SVGShapeData[];
  packedPages: PackedItem[][]; // Array of pages
  sheetWidth: number;
  sheetHeight: number;
  padding?: number; // Add padding prop for visualization
}

export const NestingCanvas: React.FC<NestingCanvasProps> = ({ shapes, packedPages, sheetWidth, sheetHeight, padding = 0 }) => {
  const [scale, setScale] = useState(0.8);
  const containerRef = useRef<HTMLDivElement>(null);
  const [imageUrls, setImageUrls] = useState<Record<string, string>>({});

  // Parse viewbox string to numbers
  const getViewBox = (vb: string) => {
    const parts = vb.split(/[\s,]+/).map(Number);
    // Standardize: x, y, width, height
    return { x: parts[0] || 0, y: parts[1] || 0, w: parts[2] || 0, h: parts[3] || 0 };
  };

  const shapeMap = useMemo(() => {
    return shapes.reduce((acc, shape) => {
      acc[shape.id] = shape;
      return acc;
    }, {} as Record<string, SVGShapeData>);
  }, [shapes]);

  // Generate Data URIs (Base64) for isolation
  // Using Base64 ensures the image data is inline and accessible when the SVG is serialized for PDF export,
  // preventing "tainted canvas" or blocked resource issues that occur with Blob URLs.
  useEffect(() => {
    const newUrls: Record<string, string> = {};
    
    shapes.forEach(shape => {
        // Construct a full SVG file string from the inner content
        const svgString = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="${shape.viewBox}">
                ${shape.svgContent}
            </svg>
        `.trim();
        
        try {
            // Encode SVG to Base64
            const base64 = window.btoa(unescape(encodeURIComponent(svgString)));
            newUrls[shape.id] = `data:image/svg+xml;base64,${base64}`;
        } catch (e) {
            console.error("Failed to encode SVG", e);
        }
    });

    setImageUrls(newUrls);
  }, [shapes]);

  // Adjust initial scale to fit screen
  useEffect(() => {
    if (containerRef.current && packedPages.length > 0) {
        const { clientWidth, clientHeight } = containerRef.current;
        const pad = 80;
        const scaleX = (clientWidth - pad) / sheetWidth;
        const scaleY = (clientHeight - pad) / sheetHeight;
        setScale(Math.min(scaleX, scaleY, 1.0)); 
    }
  }, [sheetWidth, sheetHeight, packedPages.length]);

  return (
    <div className="flex flex-col h-full bg-gray-200 rounded-lg border border-gray-300 overflow-hidden relative">
      {/* Toolbar */}
      <div className="absolute top-4 right-4 z-50 flex flex-col gap-2 bg-white/90 backdrop-blur p-2 rounded-lg shadow-lg border border-gray-200">
        <button 
          onClick={() => setScale(s => Math.min(s + 0.1, 5))}
          className="p-2 hover:bg-gray-100 rounded text-gray-700 active:bg-gray-200"
          title="Zoom In"
        >
          <ZoomIn size={20} />
        </button>
        <button 
          onClick={() => setScale(s => Math.max(s - 0.1, 0.1))}
          className="p-2 hover:bg-gray-100 rounded text-gray-700 active:bg-gray-200"
          title="Zoom Out"
        >
          <ZoomOut size={20} />
        </button>
        <button 
          onClick={() => setScale(0.8)} 
          className="p-2 hover:bg-gray-100 rounded text-gray-700 active:bg-gray-200"
          title="Reset View"
        >
          <Maximize size={20} />
        </button>
      </div>

      {/* Canvas Area */}
      <div 
        ref={containerRef} 
        className="flex-1 overflow-auto flex flex-col items-center gap-8 p-8 cursor-grab active:cursor-grabbing bg-gray-300"
      >
        {packedPages.length === 0 && (
            <div className="text-gray-500 mt-20 font-medium">Ready to nest...</div>
        )}

        {packedPages.map((items, pageIndex) => (
            <div key={pageIndex} className="relative group shadow-2xl">
                <div className="absolute -top-6 left-0 text-xs font-bold text-gray-600 uppercase flex items-center gap-2">
                    <span>Page {pageIndex + 1}</span>
                    <span className="text-gray-400 font-normal">({items.length} items)</span>
                </div>
                
                <div 
                    style={{
                        width: `${sheetWidth * scale}px`,
                        height: `${sheetHeight * scale}px`,
                        transition: 'width 0.2s, height 0.2s'
                    }}
                    className="bg-white transition-all overflow-hidden"
                    id={`page-render-${pageIndex}`}
                >
                    <svg 
                        width="100%" 
                        height="100%" 
                        viewBox={`0 0 ${sheetWidth} ${sheetHeight}`}
                        className="block overflow-hidden" 
                        preserveAspectRatio="xMidYMid meet"
                    >
                        {/* Grid */}
                        <rect x="0" y="0" width={sheetWidth} height={sheetHeight} fill="#ffffff" />
                        <defs>
                            <pattern id={`grid-${pageIndex}`} width="50" height="50" patternUnits="userSpaceOnUse">
                                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#f0f0f0" strokeWidth="0.5"/>
                            </pattern>
                        </defs>
                        <rect width="100%" height="100%" fill={`url(#grid-${pageIndex})`} />
                        <rect x="0" y="0" width={sheetWidth} height={sheetHeight} fill="none" stroke="#e5e7eb" strokeWidth="1" />

                        {items.map((item, index) => {
                            const shape = shapeMap[item.shapeId];
                            if (!shape) return null;

                            const vb = getViewBox(shape.viewBox);
                            const userScale = (shape.scale || 100) / 100;
                            const realWidthMM = shape.widthMM * userScale;
                            const realHeightMM = shape.heightMM * userScale;
                            
                            // Scale Factor: Converts BBox Units -> Millimeters
                            const scaleX = realWidthMM / shape.bbox.w;
                            const scaleY = realHeightMM / shape.bbox.h;

                            // 1. Position the Group at the center of the packed block
                            const centerX = item.x + item.width / 2;
                            const centerY = item.y + item.height / 2;

                            // 2. Rotate around the center, then shift origin to top-left of the BBox
                            const groupTransform = `
                                translate(${centerX}, ${centerY}) 
                                rotate(${item.rotation}) 
                                translate(${-realWidthMM / 2}, ${-realHeightMM / 2}) 
                            `;
                            
                            // 3. Inner Transform: 
                            // Scale from SVG units to MM
                            // Shift by negative bbox (to align bbox top-left with group origin 0,0)
                            const innerTransform = `scale(${scaleX}, ${scaleY}) translate(${-shape.bbox.x}, ${-shape.bbox.y})`;

                            return (
                                <g key={`${item.shapeId}-${index}`} transform={groupTransform}>
                                     {/* Inner Container Group */}
                                     <g transform={innerTransform}>
                                        
                                        {/* 
                                            User Content Rendered as Image via Data URI
                                        */}
                                        {imageUrls[shape.id] && (
                                            <image 
                                                href={imageUrls[shape.id]}
                                                x={vb.x}
                                                y={vb.y}
                                                width={vb.w}
                                                height={vb.h}
                                                preserveAspectRatio="none"
                                            />
                                        )}

                                        {/* 
                                            Visual Cut Line Overlay (Red)
                                            Fixed:
                                            1. strokeWidth calculated to be exactly 0.1mm in physical space (0.1 / scaleX).
                                        */}
                                        <path 
                                            d={shape.path} 
                                            style={{
                                                fill: 'none',
                                                stroke: '#ff0000',
                                                strokeWidth: `${0.1 / (scaleX || 1)}px`, 
                                                opacity: 0.8
                                            }}
                                        />
                                     </g>
                                </g>
                            );
                        })}
                    </svg>
                </div>
            </div>
        ))}
      </div>
      
      {/* Footer Info */}
      <div className="bg-white p-2 border-t text-xs text-gray-500 text-center flex justify-between px-6 z-20 font-mono">
        <span>SHEET: {sheetWidth}mm x {sheetHeight}mm</span>
        <span>PAGES: {packedPages.length}</span>
        <span>SCALE: {(scale * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
};