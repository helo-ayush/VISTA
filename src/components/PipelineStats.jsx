import React from 'react';

/**
 * Real-time timing metrics and entity count cards for each pipeline stage.
 */
const PipelineStats = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className='w-full max-w-5xl grid grid-cols-2 sm:grid-cols-5 gap-3'>
      {/* Level 1: Face Detection */}
      <div className='bg-purple-50/70 border border-purple-200 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs'>
        <span className='text-[11px] text-purple-700 font-bold uppercase tracking-wider block'>
          👤 Level 1: Face
        </span>
        <span className='text-xl font-black text-purple-950 mt-0.5'>
          {stats.faceTime} ms
        </span>
        <span className='text-[11px] text-purple-600 font-medium'>
          {stats.facesCount} face{stats.facesCount !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Level 2: Document OCR */}
      <div className='bg-emerald-50/70 border border-emerald-200 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs'>
        <div className='flex items-center justify-center gap-1'>
          <span className='text-[11px] text-emerald-700 font-bold uppercase tracking-wider'>
            🔤 Level 2: OCR
          </span>
          <span className={`text-[9px] px-1.5 py-0.2 rounded font-bold uppercase ${
            stats.providerUsed?.includes('WebGPU')
              ? 'bg-emerald-200 text-emerald-900 font-extrabold'
              : 'bg-emerald-100 text-emerald-800'
          }`}>
            {stats.providerUsed || 'WASM'}
          </span>
        </div>
        <span className='text-xl font-black text-emerald-950 mt-0.5'>
          {stats.ocrTime} ms
        </span>
        <span className='text-[11px] text-emerald-600 font-medium'>
          {stats.detectedElements} words
        </span>
      </div>

      {/* Level 3: PII Analysis */}
      <div className='bg-indigo-50/70 border border-indigo-200 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs'>
        <span className='text-[11px] text-indigo-700 font-bold uppercase tracking-wider block'>
          🛡️ Level 3: PII
        </span>
        <span className='text-xl font-black text-indigo-950 mt-0.5'>
          {stats.piiTime} ms
        </span>
        <span className='text-[11px] text-indigo-600 font-medium'>
          {stats.textEntitiesCount} sensitive
        </span>
      </div>

      {/* Level 4: Visual Redaction */}
      <div className='bg-amber-50/70 border border-amber-200 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs'>
        <span className='text-[11px] text-amber-700 font-bold uppercase tracking-wider block'>
          🎨 Level 4: Mask
        </span>
        <span className='text-xl font-black text-amber-950 mt-0.5'>
          {stats.renderTime} ms
        </span>
        <span className='text-[11px] text-amber-600 font-medium'>
          {stats.redactedCount} regions
        </span>
      </div>

      {/* Total End-to-End */}
      <div className='bg-slate-900 border border-slate-800 text-white rounded-xl p-3 text-center flex flex-col justify-center col-span-2 sm:col-span-1 shadow-2xs'>
        <span className='text-[11px] text-slate-300 font-bold uppercase tracking-wider block'>
          ⚡ Total Time
        </span>
        <span className='text-xl font-black text-white mt-0.5'>
          {stats.totalTime} ms
        </span>
        <span className='text-[11px] text-emerald-400 font-medium'>
          100% On-Device
        </span>
      </div>
    </div>
  );
};

export default PipelineStats;
