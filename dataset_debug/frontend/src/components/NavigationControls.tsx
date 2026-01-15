import React from 'react';
import { NavigationInfo } from '../types';

interface Props {
  nav: NavigationInfo;
  onNextSample: () => void;
  onPrevSample: () => void;
  onNextCoil: () => void;
  onPrevCoil: () => void;
  onNextSlice: () => void;
  onPrevSlice: () => void;
  onNextSubject: () => void;
  onPrevSubject: () => void;
  onRandom: () => void;
  onJumpToIndex: () => void;
}

export const NavigationControls: React.FC<Props> = ({
  nav, onNextSample, onPrevSample, onNextCoil, onPrevCoil,
  onNextSlice, onPrevSlice, onNextSubject, onPrevSubject,
  onRandom, onJumpToIndex
}) => {
  const btnClass = "h-5 w-5 flex items-center justify-center rounded bg-gray-800 hover:bg-gray-700 text-gray-300 disabled:opacity-30 disabled:hover:bg-gray-800 transition-colors text-xs";
  
  return (
    <div className="flex flex-wrap items-center gap-2 bg-gray-900 border border-gray-800 rounded-lg p-1.5 shadow-sm">
      <div className="flex items-center gap-1 bg-gray-950/50 rounded px-1.5 py-0.5 border border-gray-800">
        <span className="text-[10px] text-gray-500 font-bold tracking-wider mr-1">IDX</span>
        <button onClick={onPrevSample} className={btnClass}>◀</button>
        <div className="min-w-[60px] text-center font-mono text-sm font-bold text-blue-400 leading-none">
          {nav.linear_index} <span className="text-gray-600 text-[10px] font-normal">/ {nav.total_samples}</span>
        </div>
        <button onClick={onNextSample} className={btnClass}>▶</button>
      </div>

      <div className="w-px h-5 bg-gray-800"></div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Subj</span>
          <button disabled={!nav.can_prev_subject} onClick={onPrevSubject} className={btnClass}>◀</button>
          <span className="font-mono text-sm text-gray-200 min-w-[20px] text-center" title={nav.current_subject_key}>{nav.current_subject_index}</span>
          <button disabled={!nav.can_next_subject} onClick={onNextSubject} className={btnClass}>▶</button>
        </div>

        <div className="flex items-center gap-1">
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Slice</span>
          <button disabled={!nav.can_prev_slice} onClick={onPrevSlice} className={btnClass}>◀</button>
          <span className="font-mono text-sm text-gray-200 min-w-[20px] text-center">{nav.current_slice_index}</span>
          <button disabled={!nav.can_next_slice} onClick={onNextSlice} className={btnClass}>▶</button>
        </div>

        <div className={`flex items-center gap-1 ${!nav.coil_count ? 'opacity-30' : ''}`}>
          <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Coil</span>
          <button disabled={!nav.can_prev_coil} onClick={onPrevCoil} className={btnClass}>◀</button>
          <span className="font-mono text-sm text-gray-200 min-w-[20px] text-center">{nav.current_coil_index ?? '-'}</span>
          <button disabled={!nav.can_next_coil} onClick={onNextCoil} className={btnClass}>▶</button>
        </div>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button onClick={onRandom} className="px-2 py-0.5 bg-gray-800 hover:bg-gray-700 text-[11px] rounded text-gray-300 border border-gray-700">
          🎲 Rand
        </button>
        <button onClick={onJumpToIndex} className="px-2 py-0.5 bg-gray-800 hover:bg-gray-700 text-[11px] rounded text-gray-300 border border-gray-700">
          Go
        </button>
      </div>
    </div>
  );
};
