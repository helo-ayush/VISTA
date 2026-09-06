import React from 'react';

/**
 * Interactive Document Viewer Canvas with clean typographic mode toggles,
 * zoom-click inspection trigger, and image download.
 */
const DocumentCanvas = ({
  canvasRef,
  imageFile,
  viewMode,
  setViewMode,
  redactedBoxesCount,
  facesCount,
  onDownload,
  onOpenZoom
}) => {
  return (
    <div className='flex flex-col gap-3 bg-white p-4 border border-gray-200 rounded-2xl shadow-sm'>
      {/* Canvas Header & Mode Toggles */}
      <div className='flex flex-wrap items-center justify-between gap-2 border-b border-gray-100 pb-3'>
        <div className='text-sm font-bold text-gray-900 flex items-center gap-1.5'>
          <span>Document View</span>
          {imageFile && (
            <span className='text-xs text-gray-400 font-normal truncate max-w-[150px]'>
              ({imageFile.name})
            </span>
          )}
        </div>

        {/* Clean 4 View Modes (Zero Emojis) */}
        <div className='flex flex-wrap items-center bg-gray-100 p-1 rounded-xl text-xs font-semibold text-gray-600 gap-1'>
          <button
            type='button'
            onClick={() => setViewMode('guided')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              viewMode === 'guided'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'hover:text-gray-900 hover:bg-gray-200/80'
            }`}
            title='Final Protected Document with blurred faces and guided PII tags'
          >
            Protected
          </button>
          <button
            type='button'
            onClick={() => setViewMode('ocr')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              viewMode === 'ocr'
                ? 'bg-emerald-600 text-white shadow-xs'
                : 'hover:text-emerald-700 hover:bg-emerald-50'
            }`}
            title='View PaddleOCR detected text bounding boxes'
          >
            OCR Boxes
          </button>
          <button
            type='button'
            onClick={() => setViewMode('faces')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              viewMode === 'faces'
                ? 'bg-purple-600 text-white shadow-xs'
                : 'hover:text-purple-700 hover:bg-purple-50'
            }`}
            title='View YuNet Face Detection bounding boxes'
          >
            Face Boxes
          </button>
          <button
            type='button'
            onClick={() => setViewMode('original')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer ${
              viewMode === 'original'
                ? 'bg-gray-800 text-white shadow-xs'
                : 'hover:text-gray-900 hover:bg-gray-200/80'
            }`}
            title='View pristine original screenshot'
          >
            Original
          </button>
        </div>
      </div>

      {/* Canvas Viewport with Click-to-Zoom Lightbox Trigger */}
      <div
        onClick={onOpenZoom}
        className='group relative w-full flex justify-center items-center bg-slate-900/5 rounded-xl p-2 min-h-[300px] overflow-auto max-h-[550px] cursor-zoom-in transition-all hover:bg-slate-900/10'
        title='Click to open high-res zoom & pan inspector'
      >
        <canvas
          ref={canvasRef}
          className='max-w-full h-auto rounded-lg shadow-sm border border-gray-200/80 object-contain transition-transform group-hover:scale-[1.003]'
        />

        {/* Subtle Floating Zoom Pill Overlay */}
        <div className='absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity duration-200 bg-slate-900/85 text-white text-[11px] font-medium px-3 py-1.5 rounded-lg backdrop-blur-sm pointer-events-none flex items-center gap-1.5 shadow-lg'>
          <span>Click to Zoom & Inspect</span>
        </div>
      </div>

      {/* Footer Info & Action Controls */}
      <div className='flex flex-wrap items-center justify-between pt-2 border-t border-gray-100 gap-2'>
        <div className='flex items-center gap-2 text-xs text-gray-500'>
          {facesCount > 0 && (
            <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-purple-50 text-purple-700 border border-purple-200/80 font-medium'>
              {facesCount} Face{facesCount !== 1 ? 's' : ''} Blurred
            </span>
          )}
          {redactedBoxesCount > 0 ? (
            <span className='inline-flex items-center px-2 py-0.5 rounded-md bg-emerald-50 text-emerald-700 border border-emerald-200/80 font-medium'>
              {redactedBoxesCount} PII Region{redactedBoxesCount !== 1 ? 's' : ''} Masked
            </span>
          ) : (
            <span className='text-gray-400'>No sensitive PII text found</span>
          )}
        </div>

        <div className='flex items-center gap-2'>
          <button
            type='button'
            onClick={onOpenZoom}
            className='px-3 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 text-xs font-semibold rounded-lg transition-all cursor-pointer'
          >
            Zoom & Pan
          </button>
          <button
            type='button'
            onClick={onDownload}
            className='px-4 py-2 bg-gray-900 hover:bg-black text-white text-xs font-semibold rounded-lg shadow-xs transition-all cursor-pointer active:scale-95'
          >
            Download Image
          </button>
        </div>
      </div>
    </div>
  );
};

export default DocumentCanvas;
