import React from 'react';

/**
 * Interactive Document Viewer Canvas with view mode controls and image download.
 */
const DocumentCanvas = ({
  canvasRef,
  imageFile,
  viewMode,
  setViewMode,
  redactedBoxesCount,
  facesCount,
  onDownload
}) => {
  return (
    <div className='flex flex-col gap-3 bg-white p-4 border border-gray-200 rounded-2xl shadow-sm'>
      {/* Canvas Header & Mode Toggles */}
      <div className='flex flex-wrap items-center justify-between gap-2 border-b border-gray-100 pb-3'>
        <div className='text-sm font-bold text-gray-800 flex items-center gap-1.5'>
          <span>Document View</span>
          {imageFile && (
            <span className='text-xs text-gray-400 font-normal truncate max-w-[150px]'>
              ({imageFile.name})
            </span>
          )}
        </div>

        {/* Clean 4 View Modes */}
        <div className='flex flex-wrap items-center bg-gray-100 p-1 rounded-xl text-xs font-semibold text-gray-600 gap-1'>
          <button
            onClick={() => setViewMode('guided')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
              viewMode === 'guided'
                ? 'bg-indigo-600 text-white shadow-xs'
                : 'hover:text-gray-900 hover:bg-gray-200'
            }`}
            title='Final Protected Document with blurred faces and guided PII tags'
          >
            <span>🛡️</span>
            <span>Protected (Final)</span>
          </button>
          <button
            onClick={() => setViewMode('faces')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
              viewMode === 'faces'
                ? 'bg-purple-600 text-white shadow-xs'
                : 'hover:text-purple-700 hover:bg-purple-50'
            }`}
            title='View YuNet Face Detection bounding boxes'
          >
            <span>👤</span>
            <span>Face Boxes</span>
          </button>
          <button
            onClick={() => setViewMode('ocr')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
              viewMode === 'ocr'
                ? 'bg-emerald-600 text-white shadow-xs'
                : 'hover:text-emerald-700 hover:bg-emerald-50'
            }`}
            title='View PaddleOCR detected text bounding boxes'
          >
            <span>🔤</span>
            <span>OCR Boxes</span>
          </button>
          <button
            onClick={() => setViewMode('original')}
            className={`px-3 py-1.5 rounded-lg transition-all cursor-pointer flex items-center gap-1.5 ${
              viewMode === 'original'
                ? 'bg-gray-800 text-white shadow-xs'
                : 'hover:text-gray-900 hover:bg-gray-200'
            }`}
            title='View pristine original screenshot'
          >
            <span>📷</span>
            <span>Original</span>
          </button>
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className='w-full flex justify-center items-center bg-gray-900/5 rounded-xl p-2 min-h-[300px] overflow-auto max-h-[550px]'>
        <canvas
          ref={canvasRef}
          className='max-w-full h-auto rounded-lg shadow-sm border border-gray-200 object-contain'
        />
      </div>

      {/* Footer Info & Download Action */}
      <div className='flex flex-wrap items-center justify-between pt-2 border-t border-gray-100 gap-2'>
        <div className='flex items-center gap-2 text-xs text-gray-500'>
          {facesCount > 0 && (
            <span className='inline-flex items-center gap-1 px-2 py-0.5 rounded bg-purple-50 text-purple-700 border border-purple-200 font-medium'>
              👤 {facesCount} Face{facesCount !== 1 ? 's' : ''} Blurred
            </span>
          )}
          {redactedBoxesCount > 0 ? (
            <span className='inline-flex items-center gap-1 px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-medium'>
              🛡️ {redactedBoxesCount} PII Region{redactedBoxesCount !== 1 ? 's' : ''} Masked
            </span>
          ) : (
            <span>No sensitive PII text found</span>
          )}
        </div>

        <button
          onClick={onDownload}
          className='px-4 py-2 bg-gray-900 hover:bg-black text-white text-xs font-semibold rounded-lg shadow-xs flex items-center gap-1.5 cursor-pointer transition-all active:scale-95'
        >
          <svg className='w-3.5 h-3.5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
            <path strokeLinecap='round' strokeLinejoin='round' strokeWidth='2' d='M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4' />
          </svg>
          Download Redacted Image
        </button>
      </div>
    </div>
  );
};

export default DocumentCanvas;
