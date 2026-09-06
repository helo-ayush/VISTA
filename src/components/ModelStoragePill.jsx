import React, { useState } from 'react';

/**
 * Clean, artistic pill displaying client-side model storage, live download progress,
 * and one-click disk space reclamation.
 */
const ModelStoragePill = ({
  storageInfo,
  downloadProgress,
  isDownloading,
  onClearStorage,
  onPreloadModels,
  clearedMessage
}) => {
  const handleClearClick = (e) => {
    e.stopPropagation();
    onClearStorage();
  };

  // State 1: Active Downloading / Progress
  if (isDownloading && downloadProgress) {
    const pct = Math.min(Math.max(downloadProgress.progress || 0, 0), 100);
    return (
      <div className="relative overflow-hidden flex items-center gap-3 px-3.5 py-1.5 bg-white/90 backdrop-blur-md border border-indigo-200/80 rounded-full shadow-xs text-xs">
        {/* Background filling progress bar */}
        <div
          className="absolute left-0 top-0 bottom-0 bg-indigo-100/70 transition-all duration-300 ease-out -z-10"
          style={{ width: `${pct}%` }}
        />
        
        {/* Minimal rotating spinner */}
        <div className="w-3.5 h-3.5 border-2 border-indigo-600 border-t-transparent rounded-full animate-spin flex-shrink-0" />
        
        <span className="font-semibold text-indigo-900 tracking-tight">
          Downloading Models: {pct}%
        </span>

        <span className="text-[11px] font-mono text-indigo-600">
          {downloadProgress.loadedMB ? `${downloadProgress.loadedMB} MB` : `${pct}%`}
        </span>
      </div>
    );
  }

  // State 2: Recently Cleared feedback
  if (clearedMessage) {
    return (
      <div className="flex items-center gap-2 px-3.5 py-1.5 bg-emerald-50/90 backdrop-blur-md border border-emerald-200/80 rounded-full shadow-xs text-xs text-emerald-800 animate-fadeIn">
        <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
        <span className="font-medium">{clearedMessage}</span>
      </div>
    );
  }

  // State 3: Models Cached & Ready
  if (storageInfo?.isCached) {
    return (
      <div className="flex items-center gap-2.5 px-3 py-1 bg-white/80 backdrop-blur-md border border-gray-200/80 rounded-full shadow-xs text-xs text-gray-700">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse flex-shrink-0" />
        
        <span className="font-medium text-gray-800">
          Models Cached <span className="font-mono text-gray-500 text-[11px]">({storageInfo.formatted})</span>
        </span>

        <div className="h-3 w-px bg-gray-200 mx-0.5" />

        <button
          type="button"
          onClick={handleClearClick}
          className="text-[11px] font-medium text-gray-400 hover:text-red-600 px-1.5 py-0.5 rounded-md hover:bg-red-50 transition-colors cursor-pointer"
          title="Free up browser storage by deleting downloaded model weights"
        >
          Clear Space
        </button>
      </div>
    );
  }

  // State 4: Not yet downloaded / First visit
  return (
    <div className="flex items-center gap-2.5 px-3 py-1 bg-white/80 backdrop-blur-md border border-gray-200/80 rounded-full shadow-xs text-xs text-gray-600">
      <span className="w-2 h-2 rounded-full bg-amber-400 flex-shrink-0" />
      
      <span className="font-medium text-gray-600">
        Models: <span className="font-mono text-gray-400 text-[11px]">~118 MB</span>
      </span>

      <div className="h-3 w-px bg-gray-200 mx-0.5" />

      <button
        type="button"
        onClick={onPreloadModels}
        className="text-[11px] font-semibold text-indigo-600 hover:text-indigo-700 px-2 py-0.5 bg-indigo-50 hover:bg-indigo-100 rounded-full transition-all cursor-pointer active:scale-95"
        title="Preload models for instant offline redaction"
      >
        Preload
      </button>
    </div>
  );
};

export default ModelStoragePill;
