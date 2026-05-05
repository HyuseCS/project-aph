import type { PriceRecord, Stats } from '../types';

const BASE_URL = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: 'Request failed' }));
    throw new Error(error.error || `HTTP ${response.status}`);
  }

  return response.json();
}

export const api = {
  // Get all prices with optional filters
  getPrices: (filters?: {
    commodity?: string;
    start_date?: string;
    end_date?: string;
  }): Promise<{ data: PriceRecord[] }> => {
    const params = new URLSearchParams();
    if (filters?.commodity) params.set('commodity', filters.commodity);
    if (filters?.start_date) params.set('start_date', filters.start_date);
    if (filters?.end_date) params.set('end_date', filters.end_date);
    const query = params.toString();
    return request(`/prices${query ? `?${query}` : ''}`);
  },

  // Add one or more price records
  addPrices: (records: PriceRecord[]): Promise<{ message: string; added: number }> => {
    return request('/prices', {
      method: 'POST',
      body: JSON.stringify({ records }),
    });
  },

  // Update a single price record
  updatePrice: (
    date: string,
    commodity: string,
    price: number
  ): Promise<{ message: string }> => {
    return request('/prices', {
      method: 'PUT',
      body: JSON.stringify({ date, commodity, price }),
    });
  },

  // Delete a single price record
  deletePrice: (date: string, commodity: string): Promise<{ message: string }> => {
    return request('/prices', {
      method: 'DELETE',
      body: JSON.stringify({ date, commodity }),
    });
  },

  // Import CSV file
  importCSV: (file: File): Promise<{ message: string; added: number }> => {
    const formData = new FormData();
    formData.append('file', file);
    return fetch(`${BASE_URL}/prices/import`, {
      method: 'POST',
      body: formData,
    }).then((res) => {
      if (!res.ok) throw new Error('Import failed');
      return res.json();
    });
  },

  // Refresh model data
  refresh: (): Promise<{ message: string }> => {
    return request('/refresh', { method: 'POST' });
  },

  // Get stats
  getStats: (): Promise<Stats> => {
    return request('/stats');
  },
};
