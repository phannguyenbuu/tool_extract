import React, { useState, useCallback, useRef } from 'react';
import { jsPDF } from 'jspdf';
import { DEFAULT_SHEET_WIDTH, DEFAULT_SHEET_HEIGHT } from './constants';
import { RasterPacker } from './utils/packer';
import { parseSVGFile } from './utils/svgParser';
import { SVGShapeData, PackedItem, PackingResult } from './types';
import { NestingCanvas } from './components/NestingCanvas';
import { Play, RefreshCw, LayoutTemplate, Download, FileUp, Trash2, Plus, Minus, Percent, Hash, Equal, Scaling, Ruler, Scissors, BoxSelect, FileText, Grid3X3, Layers, Shuffle } from 'lucide-react';

type InputMode = 'manual' | 'ratio';
type NestingStrategy = 'mixed' | 'clustered';

const App: React.FC = () => {
  const [shapes, setShapes] = useState<SVGShapeData[]>([]);
  const [sheetSize, setSheetSize] = useState({ width: DEFAULT_SHEET_WIDTH, height: DEFAULT_SHEET_HEIGHT });
  
  const [padding, setPadding] = useState(2); 
  const [bleed, setBleed] = useState(0);    

  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<PackingResult | null>(null);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [inputMode, setInputMode] = useState<InputMode>('manual');
  const [targetTotal, setTargetTotal] = useState<number>(100);
  const [nestingStrategy, setNestingStrategy] = useState<NestingStrategy>('mixed');

  // Handle File Upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newShapes: SVGShapeData[] = [];
    for (let i = 0; i < files.length; i++) {
        try {
            const shape = await parseSVGFile(files[i]);
            if (shape) {
                if (inputMode === 'ratio' && shapes.length > 0) {
                    shape.ratio = 0; 
                    shape.quantity = 0;
                }
                newShapes.push(shape);
            }
        } catch (e) {
            console.error(`Failed to parse ${files[i].name}`, e);
            alert(`Could not parse ${files[i].name}. Make sure it is a valid SVG.`);
        }
    }
    
    setShapes(prev => [...prev, ...newShapes]);
    if (fileInputRef.current) fileInputRef.current.value = '';
    setResult(null); 
  };

  const toggleInputMode = (mode: InputMode) => {
    setInputMode(mode);
    if (mode === 'ratio') {
        const total = shapes.reduce((sum, s) => sum + s.quantity, 0);
        const base = total > 0 ? total : targetTotal;
        setTargetTotal(base);
        setShapes(prev => prev.map(s => ({
            ...s,
            ratio: total > 0 ? parseFloat(((s.quantity / total) * 100).toFixed(1)) : 0
        })));
    }
  };

  const handleTargetTotalChange = (newTotal: number) => {
    setTargetTotal(newTotal);
    setShapes(prev => prev.map(s => ({
        ...s,
        quantity: Math.round((s.ratio / 100) * newTotal)
    })));
    setResult(null);
  };

  const handleDistributeEvenly = () => {
    if (shapes.length === 0) return;
    const count = shapes.length;
    
    if (inputMode === 'ratio') {
        const evenRatio = parseFloat((100 / count).toFixed(1));
        setShapes(prev => prev.map(s => ({
            ...s,
            ratio: evenRatio,
            quantity: Math.floor(targetTotal / count)
        })));
    } else {
        const perItem = Math.max(1, Math.floor(targetTotal / count));
        setShapes(prev => prev.map(s => ({
            ...s,
            quantity: perItem,
            ratio: parseFloat(((perItem / targetTotal) * 100).toFixed(1))
        })));
    }
    setResult(null);
  };

  const handleFillSheet = () => {
      if (shapes.length === 0) return;
      const sheetArea = sheetSize.width * sheetSize.height;
      const usableArea = sheetArea * 0.85;

      const totalShapeArea = shapes.reduce((sum, s) => {
           const w = s.widthMM * (s.scale / 100) + padding;
           const h = s.heightMM * (s.scale / 100) + padding;
           return sum + (w * h);
      }, 0);
      
      if (totalShapeArea === 0) return;

      const currentTotal = shapes.reduce((sum, s) => sum + s.quantity, 0);
      
      setShapes(prev => prev.map(s => {
          const sArea = (s.widthMM * (s.scale / 100) + padding) * (s.heightMM * (s.scale / 100) + padding);
          if (shapes.length === 1) {
              return { ...s, quantity: Math.floor(usableArea / sArea) };
          }
          const share = s.quantity > 0 ? (s.quantity / currentTotal) : (1 / shapes.length);
          return { ...s, quantity: Math.floor((usableArea * share) / sArea) };
      }));
      setResult(null);
  };

  const updateQuantity = (id: string, newQ: number) => {
    const q = Math.max(0, newQ);
    setShapes(prev => prev.map(s => {
        if (s.id === id) {
            return { 
                ...s, 
                quantity: q,
                ratio: targetTotal > 0 ? parseFloat(((q / targetTotal) * 100).toFixed(1)) : 0
            };
        }
        return s;
    }));
    setResult(null);
  };

  const updateRatio = (id: string, newR: number) => {
    const r = Math.max(0, Math.min(100, newR));
    setShapes(prev => prev.map(s => {
        if (s.id === id) {
            return { 
                ...s, 
                ratio: r,
                quantity: Math.round((r / 100) * targetTotal)
            };
        }
        return s;
    }));
    setResult(null);
  };

  const updateScale = (id: string, newScale: number) => {
      const s = Math.max(1, newScale);
      setShapes(prev => prev.map(shape => 
          shape.id === id ? { ...shape, scale: s } : shape
      ));
      setResult(null);
  };

  const removeShape = (id: string) => {
      setShapes(prev => prev.filter(s => s.id !== id));
      setResult(null);
  };

  // --- Optimization Logic with Pagination ---

  const performNesting = useCallback(async () => {
    if (shapes.length === 0) return;
    
    setIsProcessing(true);
    setProgress(0);
    setResult(null);

    await new Promise(r => setTimeout(r, 50));

    const activeShapes = shapes.filter(s => s.quantity > 0);
    if (activeShapes.length === 0) {
        setIsProcessing(false);
        return;
    }

    // gapRadius is half the total padding desired between items
    const gapRadius = padding / 2;
    
    const tempPacker = new RasterPacker(sheetSize.width, sheetSize.height);
    const ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315];
    const totalPrecompute = activeShapes.length * ROTATIONS.length;
    let computedCount = 0;

    for (const shape of activeShapes) {
        for (const r of ROTATIONS) {
            await tempPacker.precomputeMask(shape, r, gapRadius);
            computedCount++;
            if (computedCount % 5 === 0) {
                 setProgress(Math.round((computedCount / totalPrecompute) * 10));
                 await new Promise(res => setTimeout(res, 0));
            }
        }
    }

    // --- STRATEGY LOGIC ---
    let packingQueue: SVGShapeData[] = [];

    if (nestingStrategy === 'mixed') {
        // Strategy 1: MIXED
        // Add all items, then sort the ENTIRE queue by size.
        // This allows small items of Type A to fill gaps in Type B.
        activeShapes.forEach(shape => {
            for(let i=0; i<shape.quantity; i++) {
                packingQueue.push(shape);
            }
        });

        // Heuristic: Sort by area descending
        packingQueue.sort((a, b) => {
            const areaA = (a.widthMM * (a.scale/100)) * (a.heightMM * (a.scale/100));
            const areaB = (b.widthMM * (b.scale/100)) * (b.heightMM * (b.scale/100));
            return areaB - areaA;
        });

    } else {
        // Strategy 2: CLUSTERED
        // Sort the shape TYPES by size first.
        // Then add all quantities of Type 1, then all of Type 2.
        // Do NOT re-sort the final queue. This forces the packer to finish Type 1 before starting Type 2.
        const sortedTypes = [...activeShapes].sort((a, b) => {
            const areaA = (a.widthMM * (a.scale/100)) * (a.heightMM * (a.scale/100));
            const areaB = (b.widthMM * (b.scale/100)) * (b.heightMM * (b.scale/100));
            return areaB - areaA;
        });

        sortedTypes.forEach(shape => {
            for(let i=0; i<shape.quantity; i++) {
                packingQueue.push(shape);
            }
        });
    }

    const totalItemsOriginal = packingQueue.length;
    const pages: PackedItem[][] = [];
    let itemsPackedCount = 0;

    let pageCount = 0;
    
    while (packingQueue.length > 0) {
        pageCount++;
        const currentPacker = new RasterPacker(sheetSize.width, sheetSize.height);
        currentPacker.maskCache = tempPacker.maskCache;

        const pageItems: PackedItem[] = [];
        const nextQueue: SVGShapeData[] = [];

        for (let i = 0; i < packingQueue.length; i++) {
            const shape = packingQueue[i];
            
            if (i % 5 === 0) {
                 await new Promise(r => setTimeout(r, 0));
                 const processed = itemsPackedCount + i;
                 setProgress(10 + Math.round((processed / totalItemsOriginal) * 90));
            }

            const item = currentPacker.insert(shape, gapRadius);
            if (item) {
                pageItems.push(item);
            } else {
                nextQueue.push(shape);
            }
        }

        if (pageItems.length === 0 && nextQueue.length > 0) {
            console.error("Stopping: Shape too big for sheet", nextQueue[0]);
            alert(`Shape "${nextQueue[0].name}" is too big to fit on the sheet!`);
            break;
        }

        pages.push(pageItems);
        itemsPackedCount += pageItems.length;
        packingQueue = nextQueue; 
        
        if (pageCount > 50) {
            alert("Max page limit reached.");
            break;
        }
    }

    const counts: Record<string, number> = {};
    pages.flat().forEach(p => counts[p.shapeId] = (counts[p.shapeId] || 0) + 1);

    setResult({
        pages,
        efficiency: 0, 
        counts,
        totalArea: sheetSize.width * sheetSize.height * pages.length,
        usedArea: 0,
        pageCount: pages.length
    });

    setProgress(100);
    setIsProcessing(false);

  }, [shapes, sheetSize, padding, bleed, nestingStrategy]);

  const handleDownloadSVG = () => {
    if (!result) return;
    const spacing = 10; 
    const totalHeight = (sheetSize.height + spacing) * result.pages.length;

    let svgContent = `<?xml version="1.0" encoding="UTF-8" standalone="no"?>
  <svg xmlns="http://www.w3.org/2000/svg" width="${sheetSize.width}mm" height="${totalHeight}mm" viewBox="0 0 ${sheetSize.width} ${totalHeight}">
    <title>Nesting Layout</title>
    <style>
      .cut-line { fill: none; stroke: #ff0000; stroke-width: 0.1mm; vector-effect: non-scaling-stroke; }
      .sheet-border { fill: none; stroke: #cccccc; stroke-width: 0.5mm; }
    </style>`;
  
    result.pages.forEach((items, pageIndex) => {
        const offsetY = pageIndex * (sheetSize.height + spacing);
        
        // Page Border
        svgContent += `<rect x="0" y="${offsetY}" width="${sheetSize.width}" height="${sheetSize.height}" class="sheet-border" />`;
        
        // Page Label
        svgContent += `<text x="5" y="${offsetY + 5}" font-family="Arial" font-size="5" fill="#aaa">Page ${pageIndex + 1}</text>`;

        items.forEach(item => {
            const shape = shapes.find(s => s.id === item.shapeId);
            if (!shape) return;
        
            // Note: We use the extracted PATH, not the full svgContent (which contains images)
            // This satisfies the requirement to only export the "Frame/Outline"
            
            const userScale = (shape.scale || 100) / 100;
            
            // Real physical dimensions of the cut path
            const realWidthMM = shape.widthMM * userScale;
            const realHeightMM = shape.heightMM * userScale;
            
            // Scaling factor from BBox units to MM
            const scaleX = realWidthMM / shape.bbox.w;
            const scaleY = realHeightMM / shape.bbox.h;
            
            // Position Calculation
            // item.x/y is top-left of the PACKED BLOCK (including padding).
            // We need to center the shape within that block.
            const centerX = item.x + item.width / 2;
            const centerY = item.y + item.height / 2 + offsetY;
        
            // Transform: Move to center -> Rotate -> Move back by half real size -> Scale -> Move back by BBox origin
            const transform = `translate(${centerX} ${centerY}) rotate(${item.rotation}) translate(${-realWidthMM / 2} ${-realHeightMM / 2}) scale(${scaleX} ${scaleY}) translate(${-shape.bbox.x} ${-shape.bbox.y})`;
            
            svgContent += `  <path d="${shape.path}" transform="${transform}" class="cut-line" />\n`;
        });
    });
  
    svgContent += '</svg>';
  
    const blob = new Blob([svgContent], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `nesting-cut-layout.svg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleExportPDF = async () => {
      if (!result) return;
      setIsProcessing(true);
      
      const doc = new jsPDF({
          orientation: sheetSize.width > sheetSize.height ? 'l' : 'p',
          unit: 'mm',
          format: [sheetSize.width, sheetSize.height]
      });

      for (let i = 0; i < result.pages.length; i++) {
          if (i > 0) doc.addPage([sheetSize.width, sheetSize.height]);

          const svgNode = document.getElementById(`page-render-${i}`)?.querySelector('svg');
          
          if (svgNode) {
              const svgData = new XMLSerializer().serializeToString(svgNode);
              const canvas = document.createElement('canvas');
              const scaleFactor = 3; 
              const pixelWidth = sheetSize.width * 3.7795 * scaleFactor; 
              const pixelHeight = sheetSize.height * 3.7795 * scaleFactor;
              
              canvas.width = pixelWidth;
              canvas.height = pixelHeight;
              const ctx = canvas.getContext('2d');
              
              if (ctx) {
                  const img = new Image();
                  const svgBlob = new Blob([svgData], {type: 'image/svg+xml;charset=utf-8'});
                  const url = URL.createObjectURL(svgBlob);
                  
                  await new Promise((resolve) => {
                      img.onload = () => {
                          ctx.fillStyle = '#ffffff';
                          ctx.fillRect(0,0, pixelWidth, pixelHeight);
                          ctx.drawImage(img, 0, 0, pixelWidth, pixelHeight);
                          resolve(null);
                      };
                      img.onerror = resolve;
                      img.src = url;
                  });
                  URL.revokeObjectURL(url);

                  const imgData = canvas.toDataURL('image/jpeg', 0.9);
                  doc.addImage(imgData, 'JPEG', 0, 0, sheetSize.width, sheetSize.height);
              }
          }
      }

      doc.save('nesting-layout.pdf');
      setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans flex flex-col md:flex-row">
      
      {/* Sidebar */}
      <aside className="w-full md:w-96 bg-white border-r border-gray-200 flex flex-col shadow-lg z-20 h-screen">
        <div className="p-6 border-b border-gray-100 flex-shrink-0">
          <div className="flex items-center gap-3 mb-4">
            <div className="bg-indigo-600 p-2 rounded-lg">
                <LayoutTemplate className="text-white" size={24} />
            </div>
            <h1 className="text-xl font-bold text-gray-900">Nesting Opt.</h1>
          </div>
          
          <div className="space-y-3">
             {/* Page Size Settings */}
            <div className="bg-indigo-50 p-3 rounded-lg border border-indigo-100">
                <div className="flex items-center gap-2 mb-2 text-indigo-700">
                    <Ruler size={14} />
                    <span className="text-xs font-bold uppercase tracking-wider">Sheet Size (mm)</span>
                </div>
                <div className="flex items-center gap-2">
                    <input 
                        type="number" 
                        value={sheetSize.width}
                        onChange={(e) => setSheetSize(s => ({...s, width: parseInt(e.target.value) || 0}))}
                        className="w-full px-2 py-1 text-sm border border-indigo-200 rounded text-center font-bold text-gray-700 outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="Width"
                    />
                    <span className="text-gray-400 font-light">x</span>
                    <input 
                        type="number" 
                        value={sheetSize.height}
                        onChange={(e) => setSheetSize(s => ({...s, height: parseInt(e.target.value) || 0}))}
                        className="w-full px-2 py-1 text-sm border border-indigo-200 rounded text-center font-bold text-gray-700 outline-none focus:ring-1 focus:ring-indigo-500"
                        placeholder="Height"
                    />
                </div>
            </div>

            {/* Bleed & Padding Settings */}
            <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 p-2 rounded-lg border border-gray-200">
                    <div className="flex items-center gap-1.5 mb-1.5 text-gray-600">
                        <BoxSelect size={14} />
                        <span className="text-[10px] font-bold uppercase">Padding (mm)</span>
                    </div>
                    <input 
                        type="number" 
                        min="0"
                        step="0.5"
                        value={padding}
                        onChange={(e) => setPadding(parseFloat(e.target.value) || 0)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded text-center font-bold outline-none focus:border-indigo-500"
                    />
                </div>
                <div className="bg-gray-50 p-2 rounded-lg border border-gray-200">
                    <div className="flex items-center gap-1.5 mb-1.5 text-gray-600">
                        <Scissors size={14} />
                        <span className="text-[10px] font-bold uppercase">Bleed (mm)</span>
                    </div>
                    <input 
                        type="number" 
                        min="0"
                        step="0.5"
                        value={bleed}
                        onChange={(e) => setBleed(parseFloat(e.target.value) || 0)}
                        className="w-full px-2 py-1 text-sm border border-gray-300 rounded text-center font-bold outline-none focus:border-indigo-500"
                    />
                </div>
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar flex flex-col">
            
            {/* Controls Section */}
            <div className="p-4 space-y-4">
                
                {/* Upload */}
                <div>
                    <input 
                        type="file" 
                        accept=".svg" 
                        multiple 
                        ref={fileInputRef}
                        className="hidden" 
                        onChange={handleFileUpload} 
                    />
                    <button 
                        onClick={() => fileInputRef.current?.click()}
                        className="w-full border-2 border-dashed border-indigo-200 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-xl p-4 flex flex-col items-center justify-center transition-colors gap-1"
                    >
                        <FileUp size={24} />
                        <span className="font-medium text-sm">Add SVG Files</span>
                    </button>
                </div>

                {shapes.length > 0 && (
                    <>
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between mb-3">
                             <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Quantities</h3>
                             <div className="flex bg-gray-200 rounded p-0.5">
                                <button 
                                    onClick={() => toggleInputMode('manual')}
                                    className={`p-1 rounded text-xs font-medium flex items-center gap-1 ${inputMode === 'manual' ? 'bg-white shadow text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                                    title="Manual Count Mode"
                                >
                                    <Hash size={12}/> Manual
                                </button>
                                <button 
                                    onClick={() => toggleInputMode('ratio')}
                                    className={`p-1 rounded text-xs font-medium flex items-center gap-1 ${inputMode === 'ratio' ? 'bg-white shadow text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                                    title="Percentage Ratio Mode"
                                >
                                    <Percent size={12}/> Ratio
                                </button>
                             </div>
                        </div>

                        <div className="space-y-2">
                             {inputMode === 'ratio' ? (
                                <div className="flex items-center justify-between gap-2">
                                    <label className="text-xs font-medium text-gray-600 whitespace-nowrap">Target Total:</label>
                                    <input 
                                        type="number" 
                                        min="1"
                                        value={targetTotal}
                                        onChange={(e) => handleTargetTotalChange(parseInt(e.target.value) || 0)}
                                        className="w-20 px-2 py-1 text-sm border border-gray-300 rounded focus:ring-1 focus:ring-indigo-500 outline-none text-right"
                                    />
                                </div>
                             ) : (
                                <button 
                                    onClick={handleFillSheet}
                                    className="w-full py-2 px-3 bg-indigo-50 border border-indigo-200 hover:bg-indigo-100 text-indigo-700 text-xs font-bold rounded flex items-center justify-center gap-2 transition-colors mb-2"
                                >
                                    <Grid3X3 size={16} />
                                    Fill Sheet (Max Fit)
                                </button>
                             )}
                             
                             <button 
                                onClick={handleDistributeEvenly}
                                className="w-full py-1.5 px-3 bg-white border border-gray-300 hover:bg-gray-50 text-gray-700 text-xs font-medium rounded flex items-center justify-center gap-2 transition-colors"
                             >
                                <Equal size={14} />
                                Distribute Evenly
                             </button>
                        </div>
                    </div>

                    {/* Strategy Toggle */}
                    <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
                        <div className="flex items-center justify-between mb-2">
                            <h3 className="text-xs font-bold text-gray-500 uppercase tracking-wider">Strategy</h3>
                        </div>
                        <div className="flex bg-gray-200 rounded p-1 w-full">
                            <button 
                                onClick={() => setNestingStrategy('mixed')}
                                className={`flex-1 py-1.5 rounded text-xs font-medium flex items-center justify-center gap-1.5 transition-all ${nestingStrategy === 'mixed' ? 'bg-white shadow text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                <Shuffle size={12}/> Mixed
                            </button>
                            <button 
                                onClick={() => setNestingStrategy('clustered')}
                                className={`flex-1 py-1.5 rounded text-xs font-medium flex items-center justify-center gap-1.5 transition-all ${nestingStrategy === 'clustered' ? 'bg-white shadow text-indigo-600' : 'text-gray-500 hover:text-gray-700'}`}
                            >
                                <Layers size={12}/> Cluster
                            </button>
                        </div>
                    </div>
                    </>
                )}
            </div>

            {/* Shape List */}
            <div className="flex-1 overflow-y-auto px-4 pb-4">
                <div className="space-y-3">
                    {shapes.map(shape => (
                        <div key={shape.id} className="bg-white border border-gray-200 rounded-lg p-2 shadow-sm flex flex-col gap-2 hover:border-indigo-300 transition-colors">
                            <div className="flex items-center gap-2">
                                {/* Preview */}
                                <div className="w-10 h-10 flex-shrink-0 bg-gray-50 rounded border flex items-center justify-center overflow-hidden relative">
                                    <svg viewBox={shape.viewBox} className="w-full h-full p-0.5">
                                        <path d={shape.path} fill={shape.color} />
                                    </svg>
                                    {shape.svgContent.includes('<image') && (
                                        <div className="absolute bottom-0 right-0 bg-white/90 rounded-tl p-0.5 shadow-sm">
                                            <FileText size={8} />
                                        </div>
                                    )}
                                </div>
                                
                                {/* Info */}
                                <div className="flex-1 min-w-0">
                                    <div className="font-medium text-xs truncate" title={shape.name}>{shape.name}</div>
                                    <div className="text-[10px] text-gray-400">
                                        Size: {Math.round(shape.widthMM)}x{Math.round(shape.heightMM)}mm
                                    </div>
                                </div>

                                <button onClick={() => removeShape(shape.id)} className="text-gray-300 hover:text-red-500 transition-colors p-1">
                                    <Trash2 size={14} />
                                </button>
                            </div>

                            {/* Controls Row */}
                            <div className="flex items-center gap-2 pt-1 border-t border-gray-50">
                                {/* Scale Input */}
                                <div className="flex items-center gap-1 bg-indigo-50 px-1.5 py-0.5 rounded text-indigo-700" title="Scale %">
                                    <Scaling size={12} />
                                    <input 
                                        type="number" 
                                        value={shape.scale}
                                        onChange={(e) => updateScale(shape.id, parseInt(e.target.value) || 100)}
                                        className="w-8 bg-transparent text-xs font-bold text-center outline-none" 
                                    />
                                    <span className="text-[9px]">%</span>
                                </div>

                                <div className="flex-1"></div>

                                {/* Qty Inputs */}
                                {inputMode === 'manual' ? (
                                    <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-0.5">
                                        <button onClick={() => updateQuantity(shape.id, shape.quantity - 1)} className="p-1 hover:bg-white rounded shadow-sm text-gray-600"><Minus size={12}/></button>
                                        <input 
                                            type="number" 
                                            value={shape.quantity} 
                                            onChange={(e) => updateQuantity(shape.id, parseInt(e.target.value) || 0)}
                                            className="w-10 text-center text-xs font-bold bg-transparent outline-none appearance-none"
                                        />
                                        <button onClick={() => updateQuantity(shape.id, shape.quantity + 1)} className="p-1 hover:bg-white rounded shadow-sm text-gray-600"><Plus size={12}/></button>
                                    </div>
                                ) : (
                                    <div className="flex items-center gap-2">
                                         <div className="flex items-center gap-1 border border-gray-200 rounded px-1">
                                            <input 
                                                type="number" 
                                                min="0"
                                                max="100"
                                                value={shape.ratio}
                                                onChange={(e) => updateRatio(shape.id, parseFloat(e.target.value) || 0)}
                                                className="w-10 text-right text-xs font-bold py-0.5 outline-none"
                                            />
                                            <span className="text-[10px] text-gray-500">%</span>
                                         </div>
                                         <div className="text-xs font-bold text-gray-700 w-6 text-center">
                                            {shape.quantity}
                                         </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                    
                    {shapes.length > 0 && (
                         <div className="text-xs text-center text-gray-400 pt-2 border-t border-dashed">
                             Total Count: <span className="font-bold text-gray-600">{shapes.reduce((a,b) => a + b.quantity, 0)}</span>
                         </div>
                    )}
                </div>
            </div>

        </div>

        {/* Bottom Actions */}
        <div className="p-4 border-t border-gray-100 bg-gray-50 flex-shrink-0 space-y-3">
             <button
              onClick={performNesting}
              disabled={isProcessing || shapes.length === 0}
              className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-lg font-medium text-white transition-all transform active:scale-95 shadow-md
                ${isProcessing || shapes.length === 0 ? 'bg-gray-400 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700'}
              `}
            >
              {isProcessing ? (
                <RefreshCw className="animate-spin" size={20} />
              ) : (
                <Play size={20} fill="currentColor" />
              )}
              {isProcessing ? `Wait... ${progress.toFixed(0)}%` : 'Start Nesting'}
            </button>

            {result && (
                <div className="space-y-2 animate-fade-in">
                    <div className="flex justify-between text-xs font-medium text-gray-600 bg-white p-2 rounded border">
                        <span>Total Pages: <span className="text-indigo-600 text-sm">{result.pageCount}</span></span>
                        <span>Items: <span className="text-green-600 text-sm">{result.pages.reduce((a,b) => a + b.length, 0)}</span></span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2">
                        <button
                            onClick={handleDownloadSVG}
                            className="flex items-center justify-center gap-1 py-2 px-2 rounded-lg font-medium text-indigo-700 bg-white border border-indigo-200 hover:bg-indigo-50 transition-colors shadow-sm text-xs"
                        >
                            <Download size={14} /> SVG
                        </button>
                        <button
                            onClick={handleExportPDF}
                            disabled={isProcessing}
                            className="flex items-center justify-center gap-1 py-2 px-2 rounded-lg font-medium text-red-700 bg-white border border-red-200 hover:bg-red-50 transition-colors shadow-sm text-xs"
                        >
                            {isProcessing ? <RefreshCw className="animate-spin" size={14} /> : <FileText size={14} />} PDF
                        </button>
                    </div>
                </div>
            )}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-2 md:p-6 h-[50vh] md:h-screen flex flex-col bg-gray-100">
        {shapes.length > 0 ? (
            <NestingCanvas shapes={shapes} packedPages={result ? result.pages : []} sheetWidth={sheetSize.width} sheetHeight={sheetSize.height} padding={padding} />
        ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-gray-400 gap-4 border-2 border-dashed border-gray-300 rounded-xl m-4">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center">
                    <LayoutTemplate size={32} className="text-gray-400"/>
                </div>
                <p>Upload SVGs to begin</p>
                <p className="text-sm opacity-60 max-w-xs text-center">Supported: Path, Rect, Circle, Polygon, Image</p>
            </div>
        )}
      </main>

    </div>
  );
};

export default App;