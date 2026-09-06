import React, { useState, useRef, useEffect, useCallback } from 'react';

/**
 * Interactive Fullscreen Lightbox with Pan & Zoom controls,
 * mouse wheel zooming, drag-to-pan, view mode switching, and image download.
 */
const ImageZoomModal = ({
  isOpen,
  onClose,
  canvasRef,
  imageFileName,
  viewMode,
  setViewMode,
  onDownload
}) => {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const containerRef = useRef(null);

  // Reset zoom & pan when opened
  useEffect(() => {
    if (isOpen) {
      setScale(1);
      setPosition({ x: 0, y: 0 });
    }
  }, [isOpen]);

  // Handle ESC key to close
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
      if (e.key === '+' || e.key === '=') setScale(s => Math.min(5, s + 0.25));
      if (e.key === '-' || e.key === '_') setScale(s => Math.max(0.5, s - 0.25));
      if (e.key === '0') {
        setScale(1);
        setPosition({ x: 0, y: 0 });
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale(prev => Math.min(5, Math.max(0.4, Number((prev * zoomFactor).toFixed(2)))));
  }, []);

  // Mouse drag pan
  const handleMouseDown = (e) => {
    if (e.button !== 0) return; // Only left click
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    setPosition({
      x: e.clientX - dragStart.x,
      y: e.clientY - dragStart.y
    });
  };

  const handleMouseUp = () => setIsDragging(false);

  // Double-click to toggle between 1x and 2.5x
  const handleDoubleClick = () => {
    if (scale > 1.2) {
      setScale(1);
      setPosition({ x: 0, y: 0 });
    } else {
      setScale(2.5);
    }
  };

  if (!isOpen) return null;

  // Render high-res snapshot of the source canvas
  const canvasDataUrl = canvasRef?.current?.toDataURL('image/png');

  return (
    <div
      className='fixed inset-0 z-50 flex flex-col bg-slate-950/90 backdrop-blur-md text-white select-none animate-in fade-in duration-200'
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Top Header & Toolbar */}
      <div className='flex items-center justify-between px-6 py-3.5 bg-slate-900/80 border-b border-slate-800/80 z-10'>
        <div className='flex items-center gap-3'>
          <h2 className='text-sm font-bold tracking-tight text-white'>
            Inspect & Zoom
          </h2>
          {imageFileName && (
            <span className='text-xs text-slate-400 font-mono hidden sm:inline'>
              ({imageFileName})
            </span>
          )}
        </div>

        {/* View Mode Switcher inside Lightbox */}
        <div className='flex items-center bg-slate-800/80 p-0.5 rounded-lg text-xs font-semibold text-slate-300 gap-0.5 border border-slate-700/60'>
          <button
            type='button'
            onClick={() => setViewMode('guided')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
              viewMode === 'guided' ? 'bg-indigo-600 text-white shadow-xs' : 'hover:text-white hover:bg-slate-700/60'
            }`}
          >
            Protected
          </button>
          <button
            type='button'
            onClick={() => setViewMode('ocr')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
              viewMode === 'ocr' ? 'bg-emerald-600 text-white shadow-xs' : 'hover:text-white hover:bg-slate-700/60'
            }`}
          >
            OCR Boxes
          </button>
          <button
            type='button'
            onClick={() => setViewMode('faces')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
              viewMode === 'faces' ? 'bg-purple-600 text-white shadow-xs' : 'hover:text-white hover:bg-slate-700/60'
            }`}
          >
            Face Boxes
          </button>
          <button
            type='button'
            onClick={() => setViewMode('original')}
            className={`px-3 py-1.5 rounded-md transition-all cursor-pointer ${
              viewMode === 'original' ? 'bg-slate-700 text-white shadow-xs' : 'hover:text-white hover:bg-slate-700/60'
            }`}
          >
            Original
          </button>
        </div>

        {/* Action Controls & Close */}
        <div className='flex items-center gap-2'>
          <button
            type='button'
            onClick={onDownload}
            className='px-3.5 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold rounded-lg transition-all cursor-pointer shadow-xs'
          >
            Download
          </button>
          <button
            type='button'
            onClick={onClose}
            className='p-1.5 hover:bg-slate-800 text-slate-400 hover:text-white rounded-lg transition-all cursor-pointer'
            title='Close (Esc)'
          >
            <svg className='w-5 h-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
              <path strokeLinecap='round' strokeLinejoin='round' strokeWidth='2' d='M6 18L18 6M6 6l12 12' />
            </svg>
          </button>
        </div>
      </div>

      {/* Main Pan/Zoom Interactive Viewport */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        className={`flex-1 overflow-hidden relative flex items-center justify-center ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
      >
        {canvasDataUrl ? (
          <img
            src={canvasDataUrl}
            alt='Zoom preview'
            draggable={false}
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transition: isDragging ? 'none' : 'transform 0.12s ease-out',
              transformOrigin: 'center center'
            }}
            className='max-w-none shadow-2xl rounded-sm pointer-events-none select-none'
          />
        ) : (
          <div className='text-slate-500 text-sm'>No image data available.</div>
        )}
      </div>

      {/* Bottom Floating Zoom Controls Bar */}
      <div className='absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3 py-2 bg-slate-900/90 border border-slate-800 rounded-xl shadow-2xl backdrop-blur-md text-xs font-semibold text-slate-200 z-10'>
        <button
          type='button'
          onClick={() => setScale(s => Math.max(0.4, Number((s - 0.25).toFixed(2))))}
          className='w-7 h-7 flex items-center justify-center rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-all cursor-pointer'
          title='Zoom Out (-)'
        >
          −
        </button>
        <span className='min-w-[54px] text-center font-mono text-slate-300 text-[11px] font-bold'>
          {Math.round(scale * 100)}%
        </span>
        <button
          type='button'
          onClick={() => setScale(s => Math.min(5, Number((s + 0.25).toFixed(2))))}
          className='w-7 h-7 flex items-center justify-center rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white transition-all cursor-pointer'
          title='Zoom In (+)'
        >
          +
        </button>
        <div className='w-px h-4 bg-slate-700 mx-1'></div>
        <button
          type='button'
          onClick={() => {
            setScale(1);
            setPosition({ x: 0, y: 0 });
          }}
          className='px-2.5 py-1 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white text-[11px] transition-all cursor-pointer'
          title='Reset View (0)'
        >
          Reset (1:1)
        </button>
        <span className='text-[10px] text-slate-500 pl-2 hidden md:inline'>
          Scroll to zoom • Drag to pan • Double-click to expand
        </span>
      </div>
    </div>
  );
};

export default ImageZoomModal;
