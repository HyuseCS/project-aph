export interface PriceRecord {
  date: string;
  commodity: string;
  price: number;
}

export interface ApiResponse<T> {
  data?: T;
  message?: string;
  error?: string;
}

export interface Stats {
  total_records: number;
  date_range: {
    start: string;
    end: string;
  };
  commodities: {
    [key: string]: {
      count: number;
      avg_price: number;
      min_price: number;
      max_price: number;
    };
  };
}

export type ToastType = 'success' | 'error' | 'info';

export interface ToastMessage {
  id: string;
  type: ToastType;
  message: string;
}

export const COMMODITIES = ['Rice', 'Tomato', 'Cabbage', 'Kamote'] as const;

export const COMMODITY_COLORS: Record<string, string> = {
  Rice: '#f59e0b',    // amber
  Tomato: '#ef4444',  // red
  Cabbage: '#22c55e', // green
  Kamote: '#a855f7',  // purple
};
