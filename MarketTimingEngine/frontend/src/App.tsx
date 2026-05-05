import { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import PriceTable from './components/PriceTable';
import EntryForm from './components/EntryForm';
import ImportModal from './components/ImportModal';
import PriceChart from './components/PriceChart';
import ToastContainer from './components/Toast';
import { useToast } from './hooks/useToast';
import { api } from './services/api';
import type { PriceRecord, Stats } from './types';
import './App.css';

function App() {
  const [records, setRecords] = useState<PriceRecord[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isImportModalOpen, setIsImportModalOpen] = useState(false);
  
  const { toasts, addToast, removeToast } = useToast();

  const loadData = useCallback(async () => {
    setIsLoading(true);
    try {
      const [pricesRes, statsRes] = await Promise.all([
        api.getPrices(),
        api.getStats()
      ]);
      if (pricesRes.data) setRecords(pricesRes.data);
      setStats(statsRes);
    } catch (err: any) {
      addToast('error', `Failed to load data: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, [addToast]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleAddPrices = async (newRecords: PriceRecord[]) => {
    try {
      const res = await api.addPrices(newRecords);
      addToast('success', res.message || `Added ${res.added} records`);
      await loadData(); // Reload to get updated stats and sorting
    } catch (err: any) {
      addToast('error', `Failed to add records: ${err.message}`);
    }
  };

  const handleUpdatePrice = async (date: string, commodity: string, price: number) => {
    try {
      const res = await api.updatePrice(date, commodity, price);
      addToast('success', res.message || 'Record updated');
      
      // Optimistic update for UI feel, but still reload in background to update stats
      setRecords(prev => prev.map(r => 
        (r.date === date && r.commodity === commodity) ? { ...r, price } : r
      ));
      
      api.getStats().then(setStats).catch(console.error);
    } catch (err: any) {
      addToast('error', `Failed to update record: ${err.message}`);
    }
  };

  const handleDeletePrice = async (date: string, commodity: string) => {
    try {
      const res = await api.deletePrice(date, commodity);
      addToast('success', res.message || 'Record deleted');
      
      setRecords(prev => prev.filter(r => !(r.date === date && r.commodity === commodity)));
      api.getStats().then(setStats).catch(console.error);
    } catch (err: any) {
      addToast('error', `Failed to delete record: ${err.message}`);
    }
  };

  const handleImportCSV = async (file: File) => {
    try {
      const res = await api.importCSV(file);
      addToast('success', res.message || `Imported ${res.added} records successfully`);
      await loadData();
    } catch (err: any) {
      addToast('error', `Failed to import CSV: ${err.message}`);
      throw err; // Re-throw so modal doesn't close on error
    }
  };

  const handleRefreshModel = async () => {
    setIsRefreshing(true);
    try {
      const res = await api.refresh();
      addToast('success', res.message || 'Model data refreshed successfully');
    } catch (err: any) {
      addToast('error', `Failed to refresh model data: ${err.message}`);
    } finally {
      setIsRefreshing(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0f] p-4 md:p-8">
      <ToastContainer toasts={toasts} onRemove={removeToast} />
      
      <div className="max-w-7xl mx-auto">
        <Header 
          stats={stats} 
          onRefresh={handleRefreshModel} 
          isRefreshing={isRefreshing} 
        />
        
        {isLoading && records.length === 0 ? (
          <div className="flex justify-center items-center h-64 glass-card">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-accent"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 space-y-6">
              <PriceChart records={records} />
              <PriceTable 
                records={records} 
                onUpdate={handleUpdatePrice} 
                onDelete={handleDeletePrice} 
              />
            </div>
            
            <div className="lg:col-span-1">
              <div className="sticky top-6 space-y-6">
                <EntryForm 
                  onSubmit={handleAddPrices} 
                  existingRecords={records} 
                />
                
                <div className="glass-card p-5 animate-fade-in" style={{ animationDelay: '0.1s' }}>
                  <h3 className="font-semibold text-text-primary mb-3">Bulk Actions</h3>
                  <button 
                    onClick={() => setIsImportModalOpen(true)}
                    className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-hover hover:bg-surface border border-border hover:border-accent/50 text-text-primary rounded-xl transition-all cursor-pointer"
                  >
                    <span>📁</span> Import CSV
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      <ImportModal 
        isOpen={isImportModalOpen} 
        onClose={() => setIsImportModalOpen(false)}
        onImport={handleImportCSV}
      />
    </div>
  );
}

export default App;
