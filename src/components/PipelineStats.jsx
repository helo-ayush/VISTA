import React from 'react';

/**
 * Clean, artistic timing metrics and stage count cards with minimal typography and zero emoji clutter.
 */
const PipelineStats = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className='w-full max-w-5xl grid grid-cols-2 sm:grid-cols-5 gap-3'>
      {/* Level 1: Face Detection */}
      <div className='bg-purple-50/60 border border-purple-200/80 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs transition-all hover:bg-purple-50'>
        <span className='text-[11px] text-purple-700 font-bold uppercase tracking-wider block'>
          Level 1: Face
        </span>
        <span className='text-xl font-black text-purple-950 mt-0.5 tracking-tight'>
          {stats.faceTime} ms
        </span>
        <span className='text-[11px] text-purple-600 font-medium'>
          {stats.facesCount} face{stats.facesCount !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Level 2: Document OCR */}
      <div className='bg-emerald-50/60 border border-emerald-200/80 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs transition-all hover:bg-emerald-50'>
        <div className='flex items-center justify-center gap-1.5'>
          <span className='text-[11px] text-emerald-700 font-bold uppercase tracking-wider'>
            Level 2: OCR
          </span>
          <span className={`text-[9px] px-1.5 py-0.5 rounded font-mono font-bold uppercase ${
            stats.providerUsed?.includes('WebGPU')
              ? 'bg-emerald-200 text-emerald-900 font-extrabold'
              : 'bg-emerald-100 text-emerald-800'
          }`}>
            {stats.providerUsed || 'WASM'}
          </span>
        </div>
        <span className='text-xl font-black text-emerald-950 mt-0.5 tracking-tight'>
          {stats.ocrTime} ms
        </span>
        <span className='text-[11px] text-emerald-600 font-medium'>
          {stats.detectedElements} words
        </span>
      </div>

      {/* Level 3: PII Analysis */}
      <div className='bg-indigo-50/60 border border-indigo-200/80 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs transition-all hover:bg-indigo-50'>
        <span className='text-[11px] text-indigo-700 font-bold uppercase tracking-wider block'>
          Level 3: PII
        </span>
        <span className='text-xl font-black text-indigo-950 mt-0.5 tracking-tight'>
          {stats.piiTime} ms
        </span>
        <span className='text-[11px] text-indigo-600 font-medium'>
          {stats.textEntitiesCount} sensitive
        </span>
      </div>

      {/* Level 4: Visual Redaction */}
      <div className='bg-amber-50/60 border border-amber-200/80 rounded-xl p-3 text-center flex flex-col justify-center shadow-2xs transition-all hover:bg-amber-50'>
        <span className='text-[11px] text-amber-700 font-bold uppercase tracking-wider block'>
          Level 4: Mask
        </span>
        <span className='text-xl font-black text-amber-950 mt-0.5 tracking-tight'>
          {stats.renderTime} ms
        </span>
        <span className='text-[11px] text-amber-600 font-medium'>
          {stats.redactedCount} regions
        </span>
      </div>

      {/* Total End-to-End */}
      <div className='bg-slate-900 border border-slate-800 text-white rounded-xl p-3 text-center flex flex-col justify-center col-span-2 sm:col-span-1 shadow-2xs transition-all hover:bg-slate-950'>
        <span className='text-[11px] text-slate-400 font-bold uppercase tracking-wider block'>
          Total Time
        </span>
        <span className='text-xl font-black text-white mt-0.5 tracking-tight'>
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
