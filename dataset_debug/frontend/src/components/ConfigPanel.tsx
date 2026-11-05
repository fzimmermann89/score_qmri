import React, { useState } from 'react';

interface Props {
  config: any;
}

export const ConfigPanel: React.FC<Props> = ({ config }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3 py-2 flex justify-between items-center bg-gray-900 hover:bg-gray-800 transition-colors"
      >
        <span className="text-sm font-bold text-gray-300 uppercase">Dataset Config</span>
        <span className="text-gray-500">{isOpen ? '▼' : '▶'}</span>
      </button>
      
      {isOpen && (
        <div className="p-3 border-t border-gray-800 bg-gray-950 overflow-x-auto">
          <pre className="text-[11px] font-mono text-green-400 leading-tight">
            {JSON.stringify(config, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
