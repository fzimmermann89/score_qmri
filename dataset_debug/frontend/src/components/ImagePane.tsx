import React from 'react';

interface ImagePaneProps {
  base64: string;
  label: string;
  range: [number, number];
  side: 'left' | 'right';
  cmap?: 'gray' | 'twilight';
  imgRef: React.RefObject<HTMLImageElement>;
  scale: number;
  offset: { x: number; y: number };
  isDragging: boolean;
  showImage: boolean;
  showPenumbra: boolean;
  showBackground: boolean;
  penumbraOverlay?: string;
  backgroundOverlay?: string;
  onWheel: (e: React.WheelEvent) => void;
  onMouseDown: (e: React.MouseEvent) => void;
  onMouseMove: (e: React.MouseEvent, imgRef: HTMLImageElement | null, side: 'left' | 'right') => void;
  onMouseUp: () => void;
  onMouseLeave: () => void;
}

const Colorbar: React.FC<{ range: [number, number]; cmap?: 'gray' | 'twilight' }> = React.memo(({ range, cmap = 'gray' }) => {
  const gradient = cmap === 'twilight' 
    ? 'linear-gradient(to top, #5e4fa2, #3288bd, #66c2a5, #abdda4, #e6f598, #fee08b, #fdae61, #f46d43, #d53e4f, #9e0142, #5e4fa2)'
    : 'linear-gradient(to top, #000, #fff)';
  
  return (
    <div className="flex flex-col items-center gap-1 px-1 py-2 shrink-0 bg-gray-950/70 border-l border-gray-800 min-w-[28px]">
      <span className="text-[11px] font-mono text-gray-300">{range[1].toFixed(2)}</span>
      <div 
        className="w-3 flex-1 rounded-sm border border-gray-700 shadow-inner" 
        style={{ background: gradient, minHeight: 140 }}
      />
      <span className="text-[11px] font-mono text-gray-300">{range[0].toFixed(2)}</span>
    </div>
  );
});

export const ImagePane: React.FC<ImagePaneProps> = ({
  base64,
  label,
  range,
  side,
  cmap = 'gray',
  imgRef,
  scale,
  offset,
  isDragging,
  showImage,
  showPenumbra,
  showBackground,
  penumbraOverlay,
  backgroundOverlay,
  onWheel,
  onMouseDown,
  onMouseMove,
  onMouseUp,
  onMouseLeave
}) => {
  return (
    <div className="flex-1 flex min-w-0 min-h-0 relative group">
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        <div className="absolute top-2 left-2 z-10 px-2 py-0.5 bg-gray-950/60 backdrop-blur-sm rounded text-[10px] font-medium text-gray-300 pointer-events-none border border-white/10">
          {label}
        </div>
        
        <div 
          className="flex-1 relative bg-gray-900 overflow-hidden cursor-crosshair"
          onWheel={onWheel}
          onMouseDown={onMouseDown}
          onMouseMove={(e) => onMouseMove(e, imgRef.current, side)}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseLeave}
        >
          <div className="absolute inset-0 flex items-center justify-center p-0">
            <div 
              className="relative inline-block"
              style={{ 
                transform: `translate(${offset.x}px, ${offset.y}px) scale(${scale})`,
                transformOrigin: 'center center',
                transition: isDragging ? 'none' : 'transform 0.15s ease-out',
                maxWidth: '100%',
                maxHeight: '100%'
              }}
            >
              <img 
                ref={imgRef}
                src={base64} 
                className={`block max-w-full max-h-full transition-opacity ${showImage ? 'opacity-100' : 'opacity-0'}`}
                style={{ imageRendering: 'pixelated' }}
                draggable={false}
                decoding="async"
              />
              
              {showPenumbra && penumbraOverlay && (
                <img 
                  src={penumbraOverlay} 
                  className="absolute inset-0 w-full h-full opacity-60 pointer-events-none"
                  style={{ imageRendering: 'pixelated' }}
                  draggable={false}
                />
              )}
              {showBackground && backgroundOverlay && (
                <img 
                  src={backgroundOverlay} 
                  className="absolute inset-0 w-full h-full opacity-60 pointer-events-none"
                  style={{ imageRendering: 'pixelated' }}
                  draggable={false}
                />
              )}
            </div>
          </div>
        </div>
      </div>
      
      <Colorbar range={range} cmap={cmap} />
    </div>
  );
};
