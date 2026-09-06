import React, { useState, useRef, useEffect, useCallback } from 'react';
import { renderCanvasOverlay } from '../modules/redactionCanvas.js';

/**
 * Minimalist Lightbox: Translucent blurred background, floating mode pills,
 * close cross on right, and interactive pan & zoom.
 */
const ImageZoomModal = ({
  isOpen,
  onClose,
  imageElement,
  redactedBoxes = [],
  allOcrBoxes = [],
  detectedFaces = [],
  viewMode = 'guided',
  setViewMode
}) => {
  const modalCanvasRef = useRef(null);
  const containerRef = useRef(null);

  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Calculate clean fit-to-screen scale on open
  useEffect(() => {
    if (isOpen && imageElement) {
      const origW = imageElement.naturalWidth || imageElement.width || 1200;
      const origH = imageElement.naturalHeight || imageElement.height || 800;
      const availW = window.innerWidth * 0.86;
      const availH = window.innerHeight * 0.82;
      const fit = Math.min(availW / origW, availH / origH, 1);
      setScale(Number(fit.toFixed(2)));
      setPosition({ x: 0, y: 0 });
    }
  }, [isOpen, imageElement]);

  // Synchronously render to canvas whenever viewMode or state changes (Fixes instant tab switching without zoom lag)
  useEffect(() => {
    if (isOpen && imageElement && modalCanvasRef.current) {
      renderCanvasOverlay(
        modalCanvasRef.current,
        imageElement,
        redactedBoxes,
        allOcrBoxes,
        detectedFaces,
        viewMode
      );
    }
  }, [isOpen, viewMode, imageElement, redactedBoxes, allOcrBoxes, detectedFaces]);

  // Keyboard shortcut: Escape to close
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Mouse wheel zoom
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale(prev => Math.min(6, Math.max(0.2, Number((prev * zoomFactor).toFixed(2)))));
  }, []);

  // Mouse drag pan
  const handleMouseDown = (e) => {
    if (e.button !== 0) return; // Left click only
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

  // Double click toggles between 1x / fit and 2.5x
  const handleDoubleClick = () => {
    if (scale > 1.2) {
      if (imageElement) {
        const origW = imageElement.naturalWidth || 1200;
        const origH = imageElement.naturalHeight || 800;
        const fit = Math.min((window.innerWidth * 0.86) / origW, (window.innerHeight * 0.82) / origH, 1);
        setScale(Number(fit.toFixed(2)));
      } else {
        setScale(1);
      }
      setPosition({ x: 0, y: 0 });
    } else {
      setScale(2.5);
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className='fixed inset-0 z-50 flex items-center justify-center bg-white/70 backdrop-blur-2xl select-none overflow-hidden animate-in fade-in duration-150'
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
    >
      {/* Floating Center Pill Navigation to switch between modes */}
      <div className='fixed top-6 left-1/2 -translate-x-1/2 z-20 flex items-center bg-white/90 p-1 rounded-full shadow-lg border border-gray-200/80 backdrop-blur-md gap-1'>
        <button
          type='button'
          onClick={() => setViewMode('guided')}
          className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
            viewMode === 'guided'
              ? 'bg-indigo-600 text-white shadow-xs'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          }`}
        >
          Protected
        </button>
        <button
          type='button'
          onClick={() => setViewMode('ocr')}
          className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
            viewMode === 'ocr'
              ? 'bg-emerald-600 text-white shadow-xs'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          }`}
        >
          OCR Boxes
        </button>
        <button
          type='button'
          onClick={() => setViewMode('faces')}
          className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
            viewMode === 'faces'
              ? 'bg-purple-600 text-white shadow-xs'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          }`}
        >
          Face Boxes
        </button>
        <button
          type='button'
          onClick={() => setViewMode('original')}
          className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all cursor-pointer ${
            viewMode === 'original'
              ? 'bg-gray-900 text-white shadow-xs'
              : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
          }`}
        >
          Original
        </button>
      </div>

      {/* Floating Close Button on Right */}
      <button
        type='button'
        onClick={onClose}
        className='fixed top-6 right-6 z-20 p-2.5 rounded-full bg-white/90 hover:bg-white text-gray-500 hover:text-gray-900 shadow-lg border border-gray-200/80 backdrop-blur-md transition-all cursor-pointer hover:scale-105 active:scale-95'
        title='Close (Esc)'
      >
        <svg className='w-5 h-5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
          <path strokeLinecap='round' strokeLinejoin='round' strokeWidth='2.2' d='M6 18L18 6M6 6l12 12' />
        </svg>
      </button>

      {/* Main Pan/Zoom Canvas Viewport */}
      <div
        ref={containerRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onDoubleClick={handleDoubleClick}
        className={`w-full h-full flex items-center justify-center ${
          isDragging ? 'cursor-grabbing' : 'cursor-grab'
        }`}
      >
        <canvas
          ref={modalCanvasRef}
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transition: isDragging ? 'none' : 'transform 0.08s ease-out',
            transformOrigin: 'center center'
          }}
          className='max-w-none shadow-2xl rounded-lg border border-gray-200/60 object-contain pointer-events-none'
        />
      </div>
    </div>
  );
};

export default ImageZoomModal;
