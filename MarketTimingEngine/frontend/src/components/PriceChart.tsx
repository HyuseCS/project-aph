import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import 'chartjs-adapter-date-fns';
import type { PriceRecord } from '../types';
import { COMMODITIES, COMMODITY_COLORS } from '../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface PriceChartProps {
  records: PriceRecord[];
}

export default function PriceChart({ records }: PriceChartProps) {
  const chartData = useMemo(() => {
    // 1. Get unique sorted dates for the X-axis
    const dates = Array.from(new Set(records.map((r) => r.date))).sort();

    // 2. Map data for each commodity
    const datasets = COMMODITIES.map((commodity) => {
      const commodityRecords = records.filter((r) => r.commodity === commodity);
      
      // Create a map for quick lookup: date -> price
      const priceMap = new Map<string, number>();
      commodityRecords.forEach(r => priceMap.set(r.date, r.price));

      // Build data array matching the 'dates' array. Use null if no data for that date.
      const data = dates.map(date => priceMap.get(date) ?? null);

      return {
        label: commodity,
        data,
        borderColor: COMMODITY_COLORS[commodity] || '#ffffff',
        backgroundColor: `${COMMODITY_COLORS[commodity]}33` || '#ffffff33',
        tension: 0.3, // smooth curves
        pointRadius: 2,
        pointHoverRadius: 5,
        borderWidth: 2,
      };
    });

    return {
      labels: dates,
      datasets,
    };
  }, [records]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#e2e8f0', // var(--color-text-primary)
          usePointStyle: true,
          pointStyle: 'circle',
          padding: 20,
          font: {
            family: "'Inter', sans-serif",
            size: 12,
          }
        },
      },
      tooltip: {
        backgroundColor: 'rgba(26, 26, 46, 0.9)', // var(--color-surface-elevated)
        titleColor: '#e2e8f0',
        bodyColor: '#e2e8f0',
        borderColor: '#2a2a40', // var(--color-border)
        borderWidth: 1,
        padding: 12,
        titleFont: { size: 13, family: "'Inter', sans-serif" },
        bodyFont: { size: 13, family: "'Inter', sans-serif" },
        callbacks: {
          label: function(context: any) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += new Intl.NumberFormat('en-PH', { style: 'currency', currency: 'PHP' }).format(context.parsed.y);
            }
            return label;
          }
        }
      },
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          unit: 'day' as const,
          tooltipFormat: 'MMM d, yyyy',
          displayFormats: {
            day: 'MMM d'
          }
        },
        grid: {
          color: '#2a2a40', // var(--color-border)
          drawBorder: false,
        },
        ticks: {
          color: '#94a3b8', // var(--color-text-secondary)
          maxTicksLimit: 15,
          font: { family: "'Inter', sans-serif" }
        },
      },
      y: {
        beginAtZero: true,
        grid: {
          color: '#2a2a40', // var(--color-border)
          drawBorder: false,
        },
        ticks: {
          color: '#94a3b8', // var(--color-text-secondary)
          callback: function(value: any) {
            return '₱' + value;
          },
          font: { family: "'Inter', sans-serif" }
        },
      },
    },
  };

  return (
    <div className="glass-card p-5 mb-6 animate-fade-in" style={{ height: '400px' }}>
      <Line data={chartData} options={options} />
    </div>
  );
}
