import { useState, useMemo } from 'react';
import type { PriceRecord } from '../types';
import { COMMODITIES, COMMODITY_COLORS } from '../types';

interface PriceTableProps {
  records: PriceRecord[];
  onUpdate: (date: string, commodity: string, price: number) => void;
  onDelete: (date: string, commodity: string) => void;
}

type SortField = 'date' | 'commodity' | 'price';
type SortDir = 'asc' | 'desc';

export default function PriceTable({ records, onUpdate, onDelete }: PriceTableProps) {
  const [filterCommodity, setFilterCommodity] = useState('');
  const [filterDateStart, setFilterDateStart] = useState('');
  const [filterDateEnd, setFilterDateEnd] = useState('');
  const [sortField, setSortField] = useState<SortField>('date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const [editPrice, setEditPrice] = useState('');
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  const filtered = useMemo(() => {
    let data = [...records];
    if (filterCommodity) data = data.filter((r) => r.commodity === filterCommodity);
    if (filterDateStart) data = data.filter((r) => r.date >= filterDateStart);
    if (filterDateEnd) data = data.filter((r) => r.date <= filterDateEnd);
    data.sort((a, b) => {
      let cmp = 0;
      if (sortField === 'date') cmp = a.date.localeCompare(b.date);
      else if (sortField === 'commodity') cmp = a.commodity.localeCompare(b.commodity);
      else cmp = a.price - b.price;
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return data;
  }, [records, filterCommodity, filterDateStart, filterDateEnd, sortField, sortDir]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortField(field); setSortDir(field === 'date' ? 'desc' : 'asc'); }
  };

  const startEdit = (r: PriceRecord) => { setEditingKey(`${r.date}|${r.commodity}`); setEditPrice(String(r.price)); };
  const saveEdit = (r: PriceRecord) => { const p = parseFloat(editPrice); if (p > 0) onUpdate(r.date, r.commodity, p); setEditingKey(null); };
  const cancelEdit = () => setEditingKey(null);
  const handleKeyDown = (e: React.KeyboardEvent, r: PriceRecord) => { if (e.key === 'Enter') saveEdit(r); if (e.key === 'Escape') cancelEdit(); };

  const confirmDelete = (r: PriceRecord) => {
    const key = `${r.date}|${r.commodity}`;
    if (deleteConfirm === key) { onDelete(r.date, r.commodity); setDeleteConfirm(null); }
    else { setDeleteConfirm(key); setTimeout(() => setDeleteConfirm(null), 3000); }
  };

  const sortIcon = (f: SortField) => sortField !== f ? '↕' : sortDir === 'asc' ? '↑' : '↓';

  return (
    <div className="glass-card mb-6 animate-fade-in">
      <div className="p-5 border-b border-border">
        <div className="flex flex-col sm:flex-row gap-3">
          <select value={filterCommodity} onChange={(e) => setFilterCommodity(e.target.value)} className="px-4 py-2 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors appearance-none cursor-pointer">
            <option value="">All Commodities</option>
            {COMMODITIES.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
          <input type="date" value={filterDateStart} onChange={(e) => setFilterDateStart(e.target.value)} className="px-4 py-2 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors" />
          <input type="date" value={filterDateEnd} onChange={(e) => setFilterDateEnd(e.target.value)} className="px-4 py-2 bg-surface border border-border rounded-xl text-text-primary focus:outline-none focus:border-border-focus transition-colors" />
          {(filterCommodity || filterDateStart || filterDateEnd) && (
            <button onClick={() => { setFilterCommodity(''); setFilterDateStart(''); setFilterDateEnd(''); }} className="px-4 py-2 text-sm text-text-muted hover:text-text-primary transition-colors cursor-pointer">Clear Filters</button>
          )}
          <span className="self-center text-text-muted text-sm ml-auto">{filtered.length} record{filtered.length !== 1 ? 's' : ''}</span>
        </div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border">
              {(['date', 'commodity', 'price'] as SortField[]).map((f) => (
                <th key={f} onClick={() => toggleSort(f)} className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider px-5 py-3 cursor-pointer hover:text-text-primary transition-colors select-none">
                  {f} <span className="text-text-muted/50">{sortIcon(f)}</span>
                </th>
              ))}
              <th className="text-right text-xs font-semibold text-text-muted uppercase tracking-wider px-5 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const key = `${r.date}|${r.commodity}`;
              const isEd = editingKey === key;
              const isDel = deleteConfirm === key;
              const color = COMMODITY_COLORS[r.commodity] || '#94a3b8';
              return (
                <tr key={key} className="border-b border-border/50 hover:bg-surface-hover/30 transition-colors group">
                  <td className="px-5 py-3 text-sm font-mono text-text-secondary">{r.date}</td>
                  <td className="px-5 py-3 text-sm font-medium text-text-primary">
                    <span className="flex items-center"><span className="inline-block w-2.5 h-2.5 rounded-full mr-2" style={{ backgroundColor: color }} />{r.commodity}</span>
                  </td>
                  <td className="px-5 py-3 text-sm">
                    {isEd ? <input type="number" step="0.01" value={editPrice} onChange={(e) => setEditPrice(e.target.value)} onKeyDown={(e) => handleKeyDown(e, r)} autoFocus className="w-28 px-3 py-1.5 bg-surface border border-border-focus rounded-lg text-text-primary focus:outline-none text-sm" />
                      : <span className="font-medium text-text-primary">₱{r.price.toFixed(2)}</span>}
                  </td>
                  <td className="px-5 py-3 text-right">
                    <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      {isEd ? (<>
                        <button onClick={() => saveEdit(r)} className="px-3 py-1 text-xs bg-success/20 text-success rounded-lg hover:bg-success/30 transition-colors cursor-pointer">Save</button>
                        <button onClick={cancelEdit} className="px-3 py-1 text-xs bg-surface-hover text-text-muted rounded-lg hover:text-text-primary transition-colors cursor-pointer">Cancel</button>
                      </>) : (<>
                        <button onClick={() => startEdit(r)} className="px-3 py-1 text-xs bg-accent/20 text-accent rounded-lg hover:bg-accent/30 transition-colors cursor-pointer">Edit</button>
                        <button onClick={() => confirmDelete(r)} className={`px-3 py-1 text-xs rounded-lg transition-colors cursor-pointer ${isDel ? 'bg-danger text-white' : 'bg-danger/20 text-danger hover:bg-danger/30'}`}>{isDel ? 'Confirm?' : 'Delete'}</button>
                      </>)}
                    </div>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && <tr><td colSpan={4} className="text-center text-text-muted py-12 text-sm">No records found.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}
