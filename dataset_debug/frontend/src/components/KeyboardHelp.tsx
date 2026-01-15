import React from 'react';

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export const KeyboardHelp: React.FC<Props> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  const sections = [
    {
      title: "Navigation (Browser-Safe)",
      items: [
        { keys: ["→", "D"], desc: "Next sample" },
        { keys: ["←", "A"], desc: "Prev sample" },
        { keys: ["↑", "W"], desc: "Next slice" },
        { keys: ["↓", "S"], desc: "Prev slice" },
        { keys: [".", ">"], desc: "Next subject" },
        { keys: [",", "<"], desc: "Prev subject" },
        { keys: ["]"], desc: "Next coil" },
        { keys: ["["], desc: "Prev coil" },
        { keys: ["R"], desc: "Random sample" },
        { keys: ["G"], desc: "Jump to index" },
        { keys: ["1-9", "0"], desc: "Select dataset 1-10" },
        { keys: ["Q-P"], desc: "Select dataset 11-20" },
      ]
    },
    {
      title: "Display",
      items: [
        { keys: ["Space"], desc: "New Augmentation Seed" },
        { keys: ["T"], desc: "Toggle Augmentation" },
        { keys: ["M"], desc: "Toggle All Masks" },
        { keys: ["P"], desc: "Toggle Penumbra" },
        { keys: ["B"], desc: "Toggle Background" },
        { keys: ["H"], desc: "Toggle Histogram" },
        { keys: ["V"], desc: "Toggle Real/Imag <-> Mag/Phase" },
        { keys: ["F"], desc: "Toggle Fullscreen" },
      ]
    },
    {
      title: "Annotations",
      items: [
        { keys: ["N"], desc: "Add new field" },
        { keys: ["Y"], desc: "Save annotations" },
        { keys: ["E"], desc: "Export to JSON" },
      ]
    }
  ];

  return (
    <div className="fixed inset-0 z-[100] bg-black/80 backdrop-blur-sm flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-xl shadow-2xl w-full max-w-4xl p-6" onClick={e => e.stopPropagation()}>
        <div className="flex justify-between items-center mb-6 border-b border-gray-800 pb-4">
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <span className="text-blue-500">⌨️</span> Keyboard Shortcuts
          </h2>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-xl">✕</button>
        </div>

        <div className="grid grid-cols-3 gap-8">
          {sections.map((section) => (
            <div key={section.title}>
              <h3 className="text-blue-400 font-bold uppercase tracking-wider text-sm mb-4">{section.title}</h3>
              <ul className="space-y-2">
                {section.items.map((item, i) => (
                  <li key={i} className="flex justify-between items-center text-sm">
                    <span className="text-gray-300">{item.desc}</span>
                    <div className="flex gap-1">
                      {item.keys.map((k, j) => (
                        <kbd key={j} className="bg-gray-800 border border-gray-600 px-1.5 py-0.5 rounded text-xs font-mono text-gray-200 min-w-[20px] text-center">
                          {k}
                        </kbd>
                      ))}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="mt-8 pt-4 border-t border-gray-800 text-center text-xs text-gray-500">
          Press <kbd className="bg-gray-800 border border-gray-700 px-1 rounded">Esc</kbd> to close
        </div>
      </div>
    </div>
  );
};
