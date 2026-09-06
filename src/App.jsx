import React, { useState, useRef, useEffect, useCallback } from 'react';
import { detectFaces } from './modules/faceDetector.js';
import { runDocumentOCR } from './modules/ocrService.js';
import { redactTextContent } from './modules/piiDetector.js';
import { findBoxesToRedact, renderCanvasOverlay } from './modules/redactionCanvas.js';
import { checkStorageUsage, clearAllModelStorage, preloadAllModels, TOTAL_MODELS_SIZE_MB } from './modules/modelManager.js';
import PipelineStats from './components/PipelineStats.jsx';
import DocumentCanvas from './components/DocumentCanvas.jsx';
import ImageZoomModal from './components/ImageZoomModal.jsx';
import ModelStoragePill from './components/ModelStoragePill.jsx';

const App = () => {
  // Input & Processing State
  const [imageFile, setImageFile] = useState(null);
  const [imageElement, setImageElement] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [currentLevel, setCurrentLevel] = useState(0); // 0: idle, 1: Face, 2: OCR, 3: PII, 4: Redaction, 5: Done
  const [statusMessage, setStatusMessage] = useState('');
  const [pipelineStats, setPipelineStats] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);
  const [isZoomOpen, setIsZoomOpen] = useState(false);

  // Model Download & Local Cache Management State
  const [storageInfo, setStorageInfo] = useState(null);
  const [downloadProgress, setDownloadProgress] = useState(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [clearedMessage, setClearedMessage] = useState('');

  // Extracted Pipeline Outputs
  const [detectedFaces, setDetectedFaces] = useState([]);
  const [allOcrBoxes, setAllOcrBoxes] = useState([]);
  const [redactedBoxes, setRedactedBoxes] = useState([]);
  const [extractedOcrText, setExtractedOcrText] = useState('');
  const [redactedOcrText, setRedactedOcrText] = useState('');
  const [detectedPiiEntities, setDetectedPiiEntities] = useState([]);

  // Canvas View Mode: 'guided' (Final) | 'faces' | 'ocr' | 'original'
  const [viewMode, setViewMode] = useState('guided');

  // Inspection Tab for Level Outputs: 'pii' | 'ocr' | 'faces'
  const [inspectorTab, setInspectorTab] = useState('pii');

  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  // Re-render canvas overlay whenever viewMode, boxes, or faces change
  useEffect(() => {
    if (imageElement && canvasRef.current) {
      renderCanvasOverlay(canvasRef.current, imageElement, redactedBoxes, allOcrBoxes, detectedFaces, viewMode);
    }
  }, [viewMode, imageElement, redactedBoxes, allOcrBoxes, detectedFaces]);

  /**
   * Main Single-Click End-to-End Pipeline:
   * Level 1: YuNet ONNX Face Detection
   * Level 2: PaddleOCR v6 Document Text Extraction
   * Level 3: Xenova BERT-NER + Deterministic Regex PII & Address Detection
   * Level 4: Guided Visual Redaction (<NAME_HIDDEN>, <FACE_HIDDEN>)
   */
  const processScreenshot = useCallback(async (fileOrBlob) => {
    if (!fileOrBlob) return;
    setProcessing(true);
    setPipelineStats(null);
    setCurrentLevel(1);
    setExtractedOcrText('');
    setRedactedOcrText('');
    setAllOcrBoxes([]);
    setRedactedBoxes([]);
    setDetectedFaces([]);
    setDetectedPiiEntities([]);
    setViewMode('guided');

    const totalStart = performance.now();

    try {
      // 0. Workspace Preparation: Load image
      setStatusMessage('Loading screenshot into workspace...');
      const imgUrl = URL.createObjectURL(fileOrBlob);
      const img = new Image();
      img.src = imgUrl;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = () => reject(new Error('Failed to load image file.'));
      });
      setImageElement(img);

      // Level 1: Face Detection (YuNet ONNX)
      setCurrentLevel(1);
      setStatusMessage('Level 1/4: Detecting Faces...');
      const faceRes = await detectFaces(img);
      const { faces, timeTaken: faceTime } = faceRes;
      setDetectedFaces(faces);

      // Level 2: Document OCR (PaddleOCR v6 WASM SIMD)
      setCurrentLevel(2);
      setStatusMessage('Level 2/4: Running Accelerated Document OCR...');
      const arrayBuffer = await fileOrBlob.arrayBuffer();
      const ocrRes = await runDocumentOCR(arrayBuffer);
      const { fullText, items: ocrItems, timeTaken: ocrTime, providerUsed } = ocrRes;

      setAllOcrBoxes(ocrItems);
      setExtractedOcrText(fullText);

      // Level 3: PII & Address Detection (Xenova BERT-NER + Regex)
      setCurrentLevel(3);
      setStatusMessage('Level 3/4: Analyzing PII & Addresses (BERT-NER + Regex)...');
      const { cleanedText, redactedEntities, count: piiCount, timeTaken: piiTime } = await redactTextContent(fullText, (item) => {
        if (item && item.progress !== undefined) {
          setIsDownloading(true);
          const pct = Math.round(item.progress);
          const loaded = item.loaded ? (item.loaded / (1024 * 1024)).toFixed(1) : ((pct / 100) * 109).toFixed(1);
          setDownloadProgress({
            progress: pct,
            stage: `Downloading BERT-NER weights (${pct}%)...`,
            loadedMB: loaded,
            totalMB: 109
          });
          setStatusMessage(`Downloading neural weights (${pct}% - ${loaded}/109 MB)...`);
        }
      });
      setIsDownloading(false);
      setDownloadProgress(null);
      checkStorageUsage().then(info => setStorageInfo(info));
      setRedactedOcrText(cleanedText);
      setDetectedPiiEntities(redactedEntities);

      // Level 4: Visual Redaction & Guided Semantic Masking
      setCurrentLevel(4);
      setStatusMessage('Level 4/4: Applying Guided Semantic Redactions (<NAME_HIDDEN>, <FACE_HIDDEN>)...');
      const toRedact = findBoxesToRedact(ocrItems, redactedEntities);
      setRedactedBoxes(toRedact);

      // Render to Canvas
      const renderTime = renderCanvasOverlay(canvasRef.current, img, toRedact, ocrItems, faces, 'guided');
      const totalTime = Math.round(performance.now() - totalStart);

      // Save complete level timings & counts
      setPipelineStats({
        faceTime,
        facesCount: faces.length,
        ocrTime,
        detectedElements: ocrItems.length,
        providerUsed: providerUsed || 'WASM',
        piiTime,
        textEntitiesCount: piiCount,
        renderTime,
        redactedCount: toRedact.length,
        totalTime
      });

      setCurrentLevel(5);
      setStatusMessage(`Completed in ${totalTime} ms! Masked ${faces.length} face(s) and ${toRedact.length} PII region(s).`);
    } catch (err) {
      console.error('Pipeline Error:', err);
      alert('Error during processing: ' + err.message);
      setStatusMessage('Failed: ' + err.message);
    } finally {
      setProcessing(false);
    }
  }, []);

  // Listen for clipboard paste (Ctrl+V anywhere)
  useEffect(() => {
    const handlePaste = (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const blob = item.getAsFile();
          if (blob) {
            const pastedFile = new File([blob], `screenshot_${Date.now()}.png`, { type: blob.type });
            setImageFile(pastedFile);
            processScreenshot(pastedFile);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', handlePaste);
    return () => window.removeEventListener('paste', handlePaste);
  }, [processScreenshot]);

  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      processScreenshot(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setImageFile(file);
      processScreenshot(file);
    }
  };

  const handleLoadSample = async () => {
    try {
      setStatusMessage('Fetching sample screenshot...');
      const response = await fetch('/sample.png');
      const blob = await response.blob();
      const file = new File([blob], 'sample_id_document.png', { type: 'image/png' });
      setImageFile(file);
      processScreenshot(file);
    } catch (err) {
      alert('Failed to load sample: ' + err.message);
    }
  };

  // Check client browser storage usage on mount
  useEffect(() => {
    checkStorageUsage().then(info => setStorageInfo(info));
  }, []);

  const handleClearStorage = async () => {
    const newInfo = await clearAllModelStorage();
    setStorageInfo(newInfo);
    setClearedMessage('Storage Cleared (0 MB)');
    setTimeout(() => setClearedMessage(''), 3500);
  };

  const handlePreloadModels = async () => {
    setIsDownloading(true);
    setDownloadProgress({ progress: 0, stage: 'Starting download...', loadedMB: 0, totalMB: TOTAL_MODELS_SIZE_MB });
    try {
      const info = await preloadAllModels((p) => {
        setDownloadProgress(p);
      });
      setStorageInfo(info);
    } catch (e) {
      alert('Preload failed: ' + e.message);
    } finally {
      setIsDownloading(false);
      setDownloadProgress(null);
    }
  };

  const downloadRedactedImage = () => {
    if (!canvasRef.current) return;
    const link = document.createElement('a');
    link.download = `protected_${imageFile?.name || 'screenshot.png'}`;
    link.href = canvasRef.current.toDataURL('image/png');
    link.click();
  };

  return (
    <div className='flex flex-col w-full items-center py-6 px-4 max-w-5xl mx-auto min-h-screen font-sans text-gray-900'>
      {/* Top Bar with Badge & Model Storage Pill */}
      <div className='w-full flex items-center justify-between gap-4 mb-4'>
        <div className='text-xs font-semibold tracking-wider uppercase text-indigo-600 bg-indigo-50/90 px-3 py-1 rounded-full border border-indigo-100/80'>
          100% In-Browser Privacy
        </div>
        <ModelStoragePill
          storageInfo={storageInfo}
          downloadProgress={downloadProgress}
          isDownloading={isDownloading}
          onClearStorage={handleClearStorage}
          onPreloadModels={handlePreloadModels}
          clearedMessage={clearedMessage}
        />
      </div>

      {/* Artistic Clean Header */}
      <div className='flex flex-col items-center gap-2 text-center mb-7'>
        <h1 className='text-3xl sm:text-4xl font-black text-gray-900 tracking-tight'>
          Screenshot Privacy Redactor
        </h1>
        <p className='text-xs sm:text-sm text-gray-500 max-w-lg'>
          On-device neural face detection, high-speed OCR, and semantic privacy masking.
        </p>
      </div>

      {/* Main Upload Dropzone */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragOver(true);
        }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={`w-full max-w-5xl flex flex-col sm:flex-row items-center justify-between p-6 bg-white border-2 border-dashed rounded-2xl transition-all gap-4 mb-6 shadow-xs ${
          isDragOver
            ? 'border-indigo-600 bg-indigo-50/50 scale-[1.01]'
            : 'border-gray-300 hover:border-indigo-400'
        }`}
      >
        <div className='flex flex-col items-center sm:items-start text-center sm:text-left gap-1'>
          <div className='text-gray-900 font-bold text-base tracking-tight'>
            Upload or Paste Screenshot
          </div>
          <p className='text-xs text-gray-500'>
            Drag & drop, click to select, or press <kbd className='px-1.5 py-0.5 bg-gray-100 border border-gray-300 rounded text-[11px] font-mono'>Ctrl+V</kbd> anywhere to paste.
          </p>
        </div>

        <div className='flex items-center gap-3'>
          <input
            ref={fileInputRef}
            type='file'
            accept='image/*'
            onChange={handleFileUpload}
            className='hidden'
          />
          <button
            type='button'
            onClick={() => fileInputRef.current?.click()}
            disabled={processing}
            className='px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white text-sm font-semibold rounded-xl shadow-xs transition-all cursor-pointer disabled:opacity-50'
          >
            Select Screenshot
          </button>
          <button
            type='button'
            onClick={handleLoadSample}
            disabled={processing}
            className='px-4 py-2.5 bg-gray-100 hover:bg-gray-200 text-gray-700 text-sm font-medium rounded-xl transition-all cursor-pointer disabled:opacity-50'
          >
            Try Sample
          </button>
        </div>
      </div>

      {/* Live Level Progress Indicator */}
      {processing && (
        <div className='w-full max-w-5xl mb-6 bg-white border border-indigo-200 rounded-2xl p-4 shadow-sm flex flex-col gap-3'>
          <div className='flex items-center justify-between text-xs font-bold text-gray-700'>
            <div className='flex items-center gap-2 text-indigo-700'>
              <svg className='animate-spin h-4 w-4 text-indigo-600' xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24'>
                <circle className='opacity-25' cx='12' cy='12' r='10' stroke='currentColor' strokeWidth='4'></circle>
                <path className='opacity-75' fill='currentColor' d='M4 12a8 8 0 018-8v8H4z'></path>
              </svg>
              <span>{statusMessage}</span>
            </div>
            <span className='text-gray-400'>Level {Math.min(currentLevel, 4)} of 4</span>
          </div>

          {/* 4-Level Progress Track */}
          <div className='grid grid-cols-4 gap-2'>
            <div className={`p-2 rounded-lg text-center text-xs font-semibold border transition-all ${
              currentLevel === 1
                ? 'bg-purple-100 border-purple-400 text-purple-900 animate-pulse'
                : currentLevel > 1 ? 'bg-purple-50 border-purple-200 text-purple-700' : 'bg-gray-50 border-gray-200 text-gray-400'
            }`}>
              1. Face Detection
            </div>
            <div className={`p-2 rounded-lg text-center text-xs font-semibold border transition-all ${
              currentLevel === 2
                ? 'bg-emerald-100 border-emerald-400 text-emerald-900 animate-pulse'
                : currentLevel > 2 ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-gray-50 border-gray-200 text-gray-400'
            }`}>
              2. Document OCR
            </div>
            <div className={`p-2 rounded-lg text-center text-xs font-semibold border transition-all ${
              currentLevel === 3
                ? 'bg-indigo-100 border-indigo-400 text-indigo-900 animate-pulse'
                : currentLevel > 3 ? 'bg-indigo-50 border-indigo-200 text-indigo-700' : 'bg-gray-50 border-gray-200 text-gray-400'
            }`}>
              3. PII Analysis
            </div>
            <div className={`p-2 rounded-lg text-center text-xs font-semibold border transition-all ${
              currentLevel === 4
                ? 'bg-amber-100 border-amber-400 text-amber-900 animate-pulse'
                : currentLevel > 4 ? 'bg-amber-50 border-amber-200 text-amber-700' : 'bg-gray-50 border-gray-200 text-gray-400'
            }`}>
              4. Redaction
            </div>
          </div>
        </div>
      )}

      {/* Level Stats Bar */}
      {pipelineStats && (
        <div className='w-full max-w-5xl mb-6'>
          <PipelineStats stats={pipelineStats} />
        </div>
      )}

      {/* Main Workspace (Canvas on Left, Level Output Inspection on Right) */}
      {imageElement && (
        <div className='w-full max-w-5xl grid grid-cols-1 lg:grid-cols-12 gap-6 items-start'>
          {/* Canvas Column (7 cols) */}
          <div className='lg:col-span-7 flex flex-col gap-4'>
            <DocumentCanvas
              canvasRef={canvasRef}
              imageFile={imageFile}
              viewMode={viewMode}
              setViewMode={setViewMode}
              redactedBoxesCount={redactedBoxes.length}
              facesCount={detectedFaces.length}
              onDownload={downloadRedactedImage}
              onOpenZoom={() => setIsZoomOpen(true)}
            />
          </div>

          {/* Level Output Inspection Column (5 cols) */}
          <div className='lg:col-span-5 flex flex-col gap-3'>
            {/* Inspector Tab Switcher (Clean, Zero Emojis) */}
            <div className='bg-gray-100 p-1 rounded-xl flex items-center gap-1 text-xs font-semibold text-gray-600'>
              <button
                type='button'
                onClick={() => setInspectorTab('pii')}
                className={`flex-1 py-2 rounded-lg transition-all cursor-pointer text-center ${
                  inspectorTab === 'pii'
                    ? 'bg-white text-indigo-700 font-bold shadow-xs'
                    : 'hover:text-gray-900'
                }`}
              >
                PII Model
              </button>
              <button
                type='button'
                onClick={() => setInspectorTab('ocr')}
                className={`flex-1 py-2 rounded-lg transition-all cursor-pointer text-center ${
                  inspectorTab === 'ocr'
                    ? 'bg-white text-emerald-700 font-bold shadow-xs'
                    : 'hover:text-gray-900'
                }`}
              >
                OCR Results
              </button>
              <button
                type='button'
                onClick={() => setInspectorTab('faces')}
                className={`flex-1 py-2 rounded-lg transition-all cursor-pointer text-center ${
                  inspectorTab === 'faces'
                    ? 'bg-white text-purple-700 font-bold shadow-xs'
                    : 'hover:text-gray-900'
                }`}
              >
                Face Boxes
              </button>
            </div>

            {/* TAB 1: PII Model Output */}
            {inspectorTab === 'pii' && (
              <div className='bg-white p-4 border border-gray-200 rounded-2xl shadow-xs flex flex-col gap-3'>
                <div className='flex justify-between items-center'>
                  <div className='text-xs font-bold text-gray-800 uppercase tracking-wider flex items-center gap-1.5'>
                    <span className='w-2 h-2 rounded-full bg-indigo-500'></span>
                    <span>Level 3: PII Model Output</span>
                  </div>
                  {redactedOcrText && (
                    <button
                      onClick={() => navigator.clipboard.writeText(redactedOcrText)}
                      className='text-xs text-indigo-600 hover:text-indigo-800 font-semibold cursor-pointer'
                    >
                      Copy Redacted Text
                    </button>
                  )}
                </div>

                {/* Redacted Text Area */}
                <textarea
                  readOnly
                  value={redactedOcrText}
                  placeholder='Redacted document text will appear here...'
                  className='w-full h-44 p-3 bg-slate-50 border border-gray-200 rounded-xl text-xs font-mono text-gray-800 outline-none resize-none shadow-inner leading-relaxed'
                />

                {/* Detected Sensitive Entities */}
                <div className='flex flex-col gap-1.5'>
                  <span className='text-[11px] font-bold text-gray-600 uppercase tracking-wider'>
                    Detected Sensitive Entities ({detectedPiiEntities.length}):
                  </span>
                  <div className='flex flex-wrap gap-1.5 max-h-36 overflow-y-auto p-1 bg-slate-50/70 border border-gray-100 rounded-xl'>
                    {detectedPiiEntities.length > 0 ? (
                      detectedPiiEntities.map((ent, idx) => (
                        <span
                          key={`pii-${idx}`}
                          className='inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-xs font-mono font-medium bg-sky-50 text-sky-800 border border-sky-200'
                          title={ent.originalValue}
                        >
                          <span className='font-bold text-[10px] uppercase text-sky-600'>&lt;{ent.tag}_HIDDEN&gt;</span>
                          <span className='truncate max-w-[130px]'>{ent.originalValue}</span>
                        </span>
                      ))
                    ) : (
                      <span className='text-xs text-gray-400 p-2 italic'>No sensitive PII entities detected.</span>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* TAB 2: OCR Results */}
            {inspectorTab === 'ocr' && (
              <div className='bg-white p-4 border border-gray-200 rounded-2xl shadow-xs flex flex-col gap-3'>
                <div className='flex justify-between items-center'>
                  <div className='text-xs font-bold text-gray-800 uppercase tracking-wider flex items-center gap-1.5'>
                    <span className='w-2 h-2 rounded-full bg-emerald-500'></span>
                    <span>Level 2: OCR Results ({allOcrBoxes.length} words)</span>
                  </div>
                  {extractedOcrText && (
                    <button
                      onClick={() => navigator.clipboard.writeText(extractedOcrText)}
                      className='text-xs text-emerald-600 hover:text-emerald-800 font-semibold cursor-pointer'
                    >
                      Copy Raw Text
                    </button>
                  )}
                </div>

                {/* Raw Extracted Text */}
                <textarea
                  readOnly
                  value={extractedOcrText}
                  placeholder='Extracted OCR text will appear here...'
                  className='w-full h-52 p-3 bg-slate-50 border border-gray-200 rounded-xl text-xs font-mono text-gray-700 outline-none resize-none shadow-inner leading-relaxed'
                />

                <div className='flex items-center justify-between text-xs text-gray-500 px-1'>
                  <span>PP-OCRv6 High Accuracy</span>
                  <button
                    onClick={() => setViewMode('ocr')}
                    className='text-xs text-emerald-600 hover:underline font-medium cursor-pointer'
                  >
                    View OCR Boxes on Canvas →
                  </button>
                </div>
              </div>
            )}

            {/* TAB 3: Face Box Data */}
            {inspectorTab === 'faces' && (
              <div className='bg-white p-4 border border-gray-200 rounded-2xl shadow-xs flex flex-col gap-3'>
                <div className='flex justify-between items-center'>
                  <div className='text-xs font-bold text-gray-800 uppercase tracking-wider flex items-center gap-1.5'>
                    <span className='w-2 h-2 rounded-full bg-purple-500'></span>
                    <span>Level 1: Face Box Data ({detectedFaces.length} detected)</span>
                  </div>
                  <button
                    onClick={() => setViewMode('faces')}
                    className='text-xs text-purple-600 hover:underline font-medium cursor-pointer'
                  >
                    View on Canvas →
                  </button>
                </div>

                <div className='flex flex-col gap-2 max-h-72 overflow-y-auto p-1'>
                  {detectedFaces.length > 0 ? (
                    detectedFaces.map((face, idx) => (
                      <div
                        key={`face-box-${idx}`}
                        className='p-3 bg-purple-50/50 border border-purple-200 rounded-xl flex flex-col gap-1.5 text-xs font-mono text-purple-950'
                      >
                        <div className='flex justify-between items-center font-bold'>
                          <span className='text-purple-700'>Face #{idx + 1}</span>
                          <span className='px-2 py-0.5 rounded bg-purple-200 text-purple-800 text-[11px]'>
                            {Math.round(face.score * 100)}% Confidence
                          </span>
                        </div>
                        <div className='text-[11px] text-gray-600 grid grid-cols-2 gap-1'>
                          <span>X: {Math.round(face.x)}px</span>
                          <span>Y: {Math.round(face.y)}px</span>
                          <span>Width: {Math.round(face.width)}px</span>
                          <span>Height: {Math.round(face.height)}px</span>
                        </div>
                        {face.landmarks && face.landmarks.length > 0 && (
                          <div className='text-[10px] text-purple-600'>
                            ✓ 5 Facial Landmarks Verified
                          </div>
                        )}
                      </div>
                    ))
                  ) : (
                    <div className='p-6 text-center text-xs text-gray-400 italic bg-slate-50 border border-gray-100 rounded-xl'>
                      No faces detected in this screenshot.
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Minimalist Floating Pan & Zoom Lightbox */}
      <ImageZoomModal
        isOpen={isZoomOpen}
        onClose={() => setIsZoomOpen(false)}
        imageElement={imageElement}
        redactedBoxes={redactedBoxes}
        allOcrBoxes={allOcrBoxes}
        detectedFaces={detectedFaces}
        viewMode={viewMode}
        setViewMode={setViewMode}
      />
    </div>
  );
};

export default App;