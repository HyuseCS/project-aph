import { useState } from 'react';
import type { PriceRecord } from '../types';
import { COMMODITIES } from '../types';

interface EntryFormProps {
  onSubmit: (records: PriceRecord[]) => void;
  existingRecords: PriceRecord[];
}

export default function EntryForm({ onSubmit, existingRecords }: EntryFormProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [date, setDate] = useState(() => new Date().toISOString().split('T')[0]);
  const [entries, setEntries] = useState<{ commodity: string; price: string }[]>([
    { commodity: 'Rice', price: '' },
  ]);

  const addRow = () => {
    // Find first commodity not yet in the entries
    const used = new Set(entries.map((e) => e.commodity));
    const next = COMMODITIES.find((c) => !used.has(c)) || COMMODITIES[0];
    setEntries([...entries, { commodity: next, price: '' }]);
  };

  const removeRow = (index: number) => {
    if (entries.length <= 1) return;
    setEntries(entries.filter((_, i) => i !== index));
  };

  const updateEntry = (index: number, field: 'commodity' | 'price', value: string) => {
    const updated = [...entries];
    updated[index] = { ...updated[index], [field]: value };
    setEntries(updated);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const records: PriceRecord[] = entries
      .filter((entry) => entry.price && parseFloat(entry.price) > 0)
      .map((entry) => ({
        date,
        commodity: entry.commodity,
        price: parseFloat(entry.price),
      }));

    if (records.length === 0) return;

    onSubmit(records);
    setEntries([{ commodity: 'Rice', price: '' }]);
  };

  const isDuplicate = (commodity: string) => {
    return existingRecords.some((r) => r.date === date && r.commodity === commodity);
  };

  const fillAllCommodities = () => {
    setEntries(COMMODITIES.map((c) => ({ commodity: c, price: '' })));
  };

  return (
    <div className="glass-card mb-6 overflow-hidden animate-fade-in">
      {/* Toggle Header */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-5 hover:bg-surface-hover/30 transition-colors cursor-pointer"
      >
        <div className="flex items-center gap-3">
          <span className="text-xl">📝</span>
          <span className="font-semibold text-text-primary">Add Price Records</span>
        </div>
        <span
          className={`text-text-muted transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        >
          ▾
        </span>
      </button>

      {/* Form Body */}
      {isOpen && (
        <form onSubmit={handleSubmit} className="px-5 pb-5 border-t border-border">
          {/* Date Picker & Fill All */}
          <div className="flex flex-col sm:flex-row gap-3 mt-4 mb-4">
            <div className="flex-1">
              <label className="block text-xs text-text-muted font-medium mb-1.5 uppercase tracking-wider">
                Date
              </label>
              <input
                type="date"
                value={date}
                onChange={(e) => setDate(e.target.value)}
                className="w-full px-4 py-2.5 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors"
              />
            </div>
            <div className="flex items-end">
              <button
                type="button"
                onClick={fillAllCommodities}
                className="px-4 py-2.5 text-sm bg-surface-hover border border-border rounded-xl text-text-secondary hover:text-text-primary hover:border-accent/50 transition-all cursor-pointer"
              >
                Fill All Commodities
              </button>
            </div>
          </div>

          {/* Entry Rows */}
          <div className="space-y-3">
            {entries.map((entry, index) => (
              <div key={index} className="flex items-center gap-3">
                <select
                  value={entry.commodity}
                  onChange={(e) => updateEntry(index, 'commodity', e.target.value)}
                  className="flex-1 px-4 py-2.5 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors appearance-none cursor-pointer"
                >
                  {COMMODITIES.map((c) => (
                    <option key={c} value={c}>
                      {c}
                    </option>
                  ))}
                </select>

                <div className="relative flex-1">
                  <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted text-sm">
                    ₱
                  </span>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    placeholder="0.00"
                    value={entry.price}
                    onChange={(e) => updateEntry(index, 'price', e.target.value)}
                    className="w-full pl-8 pr-4 py-2.5 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors"
                  />
                </div>

                {isDuplicate(entry.commodity) && (
                  <span className="text-rice text-xs whitespace-nowrap" title="Record already exists for this date and commodity">
                    ⚠ exists
                  </span>
                )}

                <button
                  type="button"
                  onClick={() => removeRow(index)}
                  disabled={entries.length <= 1}
                  className="p-2 text-text-muted hover:text-danger disabled:opacity-30 transition-colors cursor-pointer disabled:cursor-not-allowed"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3 mt-4">
            {entries.length < COMMODITIES.length && (
              <button
                type="button"
                onClick={addRow}
                className="px-4 py-2 text-sm text-accent hover:text-accent-hover border border-accent/30 hover:border-accent rounded-xl transition-all cursor-pointer"
              >
                + Add Row
              </button>
            )}
            <div className="flex-1" />
            <button
              type="submit"
              className="px-6 py-2.5 bg-accent hover:bg-accent-hover text-white font-medium rounded-xl transition-all duration-200 hover:shadow-lg hover:shadow-accent/20 cursor-pointer"
            >
              Save Records
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
