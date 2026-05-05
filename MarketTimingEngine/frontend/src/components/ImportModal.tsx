import { useState, useRef } from 'react';

interface ImportModalProps {
  isOpen: boolean;
  onClose: () => void;
  onImport: (file: File) => Promise<void>;
}

export default function ImportModal({ isOpen, onClose, onImport }: ImportModalProps) {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (selectedFile: File) => {
    setError(null);
    if (!selectedFile.name.endsWith('.csv')) {
      setError('Please upload a valid CSV file.');
      setFile(null);
      return;
    }
    setFile(selectedFile);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);
    try {
      await onImport(file);
      setFile(null);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to import CSV');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-surface border border-border rounded-2xl w-full max-w-md shadow-2xl overflow-hidden">
        <div className="flex items-center justify-between p-5 border-b border-border">
          <h2 className="text-xl font-semibold text-text-primary">Import CSV</h2>
          <button
            onClick={onClose}
            className="text-text-muted hover:text-text-primary transition-colors text-xl leading-none cursor-pointer"
          >
            &times;
          </button>
        </div>

        <div className="p-6">
          <p className="text-sm text-text-secondary mb-4">
            Upload a CSV file containing market prices. The file must have the columns:{' '}
            <code className="text-accent bg-accent/10 px-1.5 py-0.5 rounded">date</code>,{' '}
            <code className="text-accent bg-accent/10 px-1.5 py-0.5 rounded">commodity</code>, and{' '}
            <code className="text-accent bg-accent/10 px-1.5 py-0.5 rounded">price</code>.
          </p>

          <div
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              dragActive
                ? 'border-accent bg-accent/5'
                : file
                ? 'border-success/50 bg-success/5'
                : 'border-border hover:border-text-muted'
            }`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={handleChange}
            />

            {file ? (
              <div>
                <div className="text-3xl mb-2">📄</div>
                <p className="font-medium text-success">{file.name}</p>
                <p className="text-xs text-text-muted mt-1">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
            ) : (
              <div className="cursor-pointer">
                <div className="text-3xl mb-2 text-text-muted">📁</div>
                <p className="font-medium text-text-primary">Drag & drop your CSV here</p>
                <p className="text-xs text-text-muted mt-1">or click to browse</p>
              </div>
            )}
          </div>

          {error && <p className="text-danger text-sm mt-4">{error}</p>}
        </div>

        <div className="flex items-center justify-end gap-3 p-5 border-t border-border bg-surface-elevated">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-text-secondary hover:text-text-primary transition-colors cursor-pointer"
            disabled={isUploading}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!file || isUploading}
            className="px-5 py-2 text-sm font-medium bg-accent hover:bg-accent-hover disabled:opacity-50 text-white rounded-lg transition-colors flex items-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {isUploading && <span className="animate-spin text-sm">⟳</span>}
            {isUploading ? 'Importing...' : 'Import Data'}
          </button>
        </div>
      </div>
    </div>
  );
}
