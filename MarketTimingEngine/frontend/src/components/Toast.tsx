import type { ToastMessage } from '../types';

interface ToastContainerProps {
  toasts: ToastMessage[];
  onRemove: (id: string) => void;
}

const typeStyles: Record<string, string> = {
  success: 'border-success/40 bg-success/10 text-success',
  error: 'border-danger/40 bg-danger/10 text-danger',
  info: 'border-info/40 bg-info/10 text-info',
};

const typeIcons: Record<string, string> = {
  success: '✓',
  error: '✕',
  info: 'ℹ',
};

export default function ToastContainer({ toasts, onRemove }: ToastContainerProps) {
  if (toasts.length === 0) return null;

  return (
    <div className="fixed top-6 right-6 z-50 flex flex-col gap-3 max-w-sm">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`animate-slide-in flex items-center gap-3 px-4 py-3 rounded-xl border backdrop-blur-md cursor-pointer transition-all duration-200 hover:scale-[1.02] ${typeStyles[toast.type]}`}
          onClick={() => onRemove(toast.id)}
        >
          <span className="text-lg font-bold">{typeIcons[toast.type]}</span>
          <span className="text-sm font-medium text-text-primary">{toast.message}</span>
        </div>
      ))}
    </div>
  );
}
