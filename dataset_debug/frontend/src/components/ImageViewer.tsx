import React, { useState, useEffect, useRef } from 'react';
import { SampleResponse, DisplayMode } from '../types';
import { ImagePane } from './ImagePane';

interface Props {
  sample: SampleResponse;
  showPenumbra: boolean;
  showBackground: boolean;
  isFullscreen: boolean;
  onCloseFullscreen: () => void;
  onIntensityChange?: (vmin: number, vmax: number) => void;
  displayMode: DisplayMode;
  onDisplayModeChange: (mode: DisplayMode) => void;
  showImage: boolean;
  onToggleImage: () => void;
  onTogglePenumbra: () => void;
  onToggleBackground: () => void;
}

export const ImageViewer: React.FC<Props> = ({ 
  sample, showPenumbra, showBackground, isFullscreen, onCloseFullscreen, onIntensityChange,
  displayMode, onDisplayModeChange, showImage, onToggleImage, onTogglePenumbra, onToggleBackground
}) => {
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoverInfo, setHoverInfo] = useState<{ x: number, y: number, value?: string, clientX: number, clientY: number } | null>(null);
  const leftImageRef = useRef<HTMLImageElement>(null);
  const rightImageRef = useRef<HTMLImageElement>(null);

  const [vmin, setVmin] = useState<number>(sample.colorbar_range.left[0]);
  const [vmax, setVmax] = useState<number>(sample.colorbar_range.left[1]);

  useEffect(() => {
    setVmin(sample.colorbar_range.left[0]);
    setVmax(sample.colorbar_range.left[1]);
  }, [sample]);

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = -e.deltaY;
    const factor = delta > 0 ? 1.1 : 0.9;
    setScale(s => Math.max(0.5, Math.min(10, s * factor)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setOffset({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleImageMouseMove = (e: React.MouseEvent, imgRef: HTMLImageElement | null, side: 'left' | 'right') => {
    if (!imgRef || isDragging) return;
    
    const rect = imgRef.getBoundingClientRect();
    const [height, width] = sample.metadata.shape;
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * width);
    const y = Math.floor(((e.clientY - rect.top) / rect.height) * height);

    if (x >= 0 && x < width && y >= 0 && y < height) {
      let value = "N/A";
      const raw = side === 'left' ? sample.raw_data_left : sample.raw_data_right;
      if (raw) {
        const pixelIndex = y * width + x;
        if (raw[pixelIndex] !== undefined) value = raw[pixelIndex].toFixed(4);
      }
      setHoverInfo({ x, y, value, clientX: e.clientX, clientY: e.clientY });
    } else {
      setHoverInfo(null);
    }
  };

  const onPaneMouseMove = (e: React.MouseEvent, imgRef: HTMLImageElement | null, side: 'left' | 'right') => {
    handleMouseMove(e);
    handleImageMouseMove(e, imgRef, side);
  };

  const containerClass = isFullscreen 
    ? "fixed inset-0 z-50 bg-gray-950 flex flex-col" 
    : "flex flex-col h-full bg-gray-950";

  return (
    <div className={containerClass}>
      <div className="flex-1 flex min-h-0 divide-x divide-gray-800 relative">
        <ImagePane
          base64={sample.image_left_base64}
          label={sample.image_left_label}
          range={sample.colorbar_range.left}
          side="left"
          cmap="gray"
          imgRef={leftImageRef}
          scale={scale}
          offset={offset}
          isDragging={isDragging}
          showImage={showImage}
          showPenumbra={showPenumbra}
          showBackground={showBackground}
          penumbraOverlay={sample.penumbra_overlay_base64}
          backgroundOverlay={sample.background_overlay_base64}
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={onPaneMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { handleMouseUp(); setHoverInfo(null); }}
        />
        
        {sample.image_right_base64 && sample.image_right_label && (
          <ImagePane
            base64={sample.image_right_base64}
            label={sample.image_right_label}
            range={sample.colorbar_range.right}
            side="right"
            cmap={sample.image_right_label === 'Phase' ? 'twilight' : 'gray'}
            imgRef={rightImageRef}
            scale={scale}
            offset={offset}
            isDragging={isDragging}
            showImage={showImage}
            showPenumbra={showPenumbra}
            showBackground={showBackground}
            penumbraOverlay={sample.penumbra_overlay_base64}
            backgroundOverlay={sample.background_overlay_base64}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={onPaneMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { handleMouseUp(); setHoverInfo(null); }}
          />
        )}

        <div className="absolute bottom-4 right-10 flex gap-2 bg-gray-900/90 backdrop-blur border border-gray-700 p-1.5 rounded-lg shadow-xl z-20 items-end">
            <div className="flex flex-col gap-1 items-center border-r border-gray-700 pr-2 mr-1">
               <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider">Mode</span>
               <div className="flex flex-col gap-1 w-full">
                  <button 
                     onClick={() => onDisplayModeChange('real_imag')}
                     className={`text-[9px] px-2 py-0.5 rounded border transition-colors ${displayMode === 'real_imag' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-800 border-gray-600 text-gray-400 hover:text-gray-300'}`}
                  >
                     Re/Im
                  </button>
                  <button 
                     onClick={() => onDisplayModeChange('mag_phase')}
                     className={`text-[9px] px-2 py-0.5 rounded border transition-colors ${displayMode === 'mag_phase' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-800 border-gray-600 text-gray-400 hover:text-gray-300'}`}
                  >
                     Mg/Ph
                  </button>
               </div>
            </div>

            <div className="flex flex-col gap-1 items-center border-r border-gray-700 pr-2 mr-1">
               <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider">Layers</span>
               <div className="flex gap-1">
                  <button 
                     onClick={onToggleImage}
                     className={`w-6 h-6 flex items-center justify-center text-[10px] rounded border transition-colors font-bold ${showImage ? 'bg-blue-600 border-blue-500 text-white' : 'bg-gray-800 border-gray-600 text-gray-500'}`}
                     title="Show Image"
                  >
                     Img
                  </button>
                  <button 
                     onClick={onTogglePenumbra}
                     className={`w-6 h-6 flex items-center justify-center text-[10px] rounded border transition-colors font-bold ${showPenumbra ? 'bg-purple-600 border-purple-500 text-white' : 'bg-gray-800 border-gray-600 text-gray-500'}`}
                     title="Show Penumbra"
                  >
                     Pen
                  </button>
                  <button 
                     onClick={onToggleBackground}
                     className={`w-6 h-6 flex items-center justify-center text-[10px] rounded border transition-colors font-bold ${showBackground ? 'bg-green-600 border-green-500 text-white' : 'bg-gray-800 border-gray-600 text-gray-500'}`}
                     title="Show Background"
                  >
                     Bg
                  </button>
               </div>
            </div>


             <div className="flex flex-col gap-1 items-center border-r border-gray-700 pr-2 mr-1">
                <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider">Zoom</span>
                <div className="flex items-center gap-1">
                   <button onClick={() => setScale(s => Math.max(0.5, s / 1.2))} className="w-5 h-5 flex items-center justify-center bg-gray-800 hover:bg-gray-700 rounded text-gray-200 transition-colors border border-gray-600">-</button>
                   <span className="text-[9px] font-mono text-blue-400 min-w-[24px] text-center">{Math.round(scale * 100)}%</span>
                   <button onClick={() => setScale(s => Math.min(10, s * 1.2))} className="w-5 h-5 flex items-center justify-center bg-gray-800 hover:bg-gray-700 rounded text-gray-200 transition-colors border border-gray-600">+</button>
                </div>
                <button onClick={() => { setScale(1); setOffset({x:0,y:0}); }} className="text-[9px] text-gray-400 hover:text-white w-full text-center hover:underline decoration-gray-500">Reset</button>
             </div>

             <div className="flex flex-col gap-1 items-center">
                <span className="text-[9px] text-gray-500 font-bold uppercase tracking-wider">Range</span>
                <div className="flex items-center gap-1">
                    <input 
                      type="number" value={vmin} onChange={e => setVmin(parseFloat(e.target.value))} 
                      className="w-10 bg-gray-950 text-[9px] p-0.5 rounded border border-gray-600 text-center text-gray-200 focus:border-blue-500 outline-none"
                      step={0.1}
                    />
                    <span className="text-gray-600 text-[9px]">-</span>
                    <input 
                      type="number" value={vmax} onChange={e => setVmax(parseFloat(e.target.value))} 
                      className="w-10 bg-gray-950 text-[9px] p-0.5 rounded border border-gray-600 text-center text-gray-200 focus:border-blue-500 outline-none"
                      step={0.1}
                    />
                </div>
                <button 
                  onClick={() => onIntensityChange?.(vmin, vmax)}
                  className="w-full bg-blue-600 hover:bg-blue-500 text-white text-[9px] py-0.5 rounded mt-0.5 font-bold shadow-sm transition-colors"
                >
                  Apply
                </button>
             </div>

             {isFullscreen && (
                <div className="border-l border-gray-700 pl-2 ml-1 flex items-center">
                   <button onClick={onCloseFullscreen} className="text-[9px] text-red-400 hover:text-red-300 border border-red-900/30 rounded px-1.5 py-0.5 hover:bg-red-900/20 transition-colors">
                      Exit
                   </button>
                </div>
             )}
        </div>
      </div>

      {hoverInfo && (

        <div 
          className="fixed z-50 pointer-events-none bg-gray-900/95 backdrop-blur-sm border border-gray-700 px-2.5 py-1.5 rounded-lg shadow-xl text-xs font-mono text-white"
          style={{ left: hoverInfo.clientX + 12, top: hoverInfo.clientY + 12 }}
        >
          <div className="flex gap-3">
            <span className="text-gray-400">x: <span className="text-white">{hoverInfo.x}</span></span>
            <span className="text-gray-400">y: <span className="text-white">{hoverInfo.y}</span></span>
          </div>
          {hoverInfo.value && (
            <div className="text-blue-400 mt-0.5">val: {hoverInfo.value}</div>
          )}
        </div>
      )}
    </div>
  );
};
