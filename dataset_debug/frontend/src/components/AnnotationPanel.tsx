import React, { useState } from 'react';
import { AnnotationField, AnnotationValues } from '../types';

interface Props {
  datasetName: string;
  fields: AnnotationField[];
  values: AnnotationValues;
  onAddField: (field: AnnotationField) => void;
  onRemoveField: (id: string) => void;
  onUpdateValue: (fieldId: string, value: any) => void;
  onExport: () => void;
}

export const AnnotationPanel: React.FC<Props> = ({
  datasetName, fields, values,
  onAddField, onRemoveField, onUpdateValue, onExport
}) => {
  const [isAdding, setIsAdding] = useState(false);
  const [newField, setNewField] = useState<AnnotationField>({ id: '', label: '', field_type: 'text' });

  const handleAddSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (newField.id && newField.label) {
      onAddField(newField);
      setIsAdding(false);
      setNewField({ id: '', label: '', field_type: 'text' });
    }
  };

  const renderInput = (field: AnnotationField) => {
    const val = values[field.id];
    switch (field.field_type) {
      case 'bool':
        return (
            <input 
              type="checkbox" 
              checked={!!val} 
              onChange={(e) => onUpdateValue(field.id, e.target.checked)}
              className="h-4 w-4 rounded border-gray-700 bg-gray-800 text-blue-500 focus:ring-blue-500"
            />
        );
      case 'number':
        return (
            <input 
              type="number" 
              value={val ?? ''} 
              onChange={(e) => {
                const val = e.target.value;
                onUpdateValue(field.id, val === '' ? null : parseFloat(val));
              }}
              className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-sm"
            />
        );
      case 'text':
      default:
        return (
          <input 
            type="text" 
            value={val ?? ''} 
            onChange={(e) => onUpdateValue(field.id, e.target.value)}
            className="w-full bg-gray-950 border border-gray-700 rounded px-2 py-1 text-xs"
          />
        );
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-3 flex flex-col h-full gap-3">
      <div className="flex justify-between items-start">
        <div className="min-w-0">
          <h3 className="text-sm font-bold text-white">Annotations</h3>
          <div className="text-[11px] text-gray-500 truncate">{datasetName}</div>
        </div>
        <button onClick={onExport} className="text-[11px] bg-green-900/80 text-green-200 px-2 py-1 rounded hover:bg-green-800">
          Export JSON
        </button>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 min-h-0">
        {fields.map(field => (
          <div key={field.id} className="group">
            <div className="flex justify-between text-sm text-gray-400 mb-1">
              <label>{field.label}</label>
              <button 
                onClick={() => onRemoveField(field.id)}
                className="opacity-0 group-hover:opacity-100 text-red-500 hover:text-red-400"
              >
                ✕
              </button>
            </div>
            {renderInput(field)}
          </div>
        ))}

        {fields.length === 0 && !isAdding && (
          <div className="text-sm text-gray-600 text-center py-6">No fields defined</div>
        )}
      </div>

      <div className="pt-3 border-t border-gray-800">
        {!isAdding ? (
          <button 
            onClick={() => setIsAdding(true)}
            className="w-full py-1 text-xs border border-dashed border-gray-700 text-gray-500 hover:border-gray-500 hover:text-gray-300 rounded"
          >
            + Add Field
          </button>
        ) : (
          <form onSubmit={handleAddSubmit} className="space-y-2 bg-gray-950 p-2 rounded border border-gray-800">
            <input 
              placeholder="ID (key)" 
              value={newField.id}
              onChange={e => setNewField({...newField, id: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs"
              required
            />
            <input 
              placeholder="Label" 
              value={newField.label}
              onChange={e => setNewField({...newField, label: e.target.value})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs"
              required
            />
            <select
              value={newField.field_type}
              onChange={e => setNewField({...newField, field_type: e.target.value as any})}
              className="w-full bg-gray-900 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300"
            >
              <option value="text">Text</option>
              <option value="number">Number</option>
              <option value="bool">Boolean</option>
            </select>
            <div className="flex gap-2">
              <button type="submit" className="flex-1 bg-blue-900 text-blue-200 text-xs py-1 rounded">Add</button>
              <button type="button" onClick={() => setIsAdding(false)} className="flex-1 bg-gray-800 text-gray-400 text-xs py-1 rounded">Cancel</button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
};
