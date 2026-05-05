import type { Stats } from '../types';

interface HeaderProps {
  stats: Stats | null;
  onRefresh: () => void;
  isRefreshing: boolean;
}

export default function Header({ stats, onRefresh, isRefreshing }: HeaderProps) {
  return (
    <header className="glass-card p-6 mb-6 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Title & Description */}
        <div>
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-accent to-kamote bg-clip-text text-transparent">
            Market Data Dashboard
          </h1>
          <p className="text-text-secondary text-sm mt-1">
            Manage commodity price data for the Market Timing Engine
          </p>
        </div>

        {/* Refresh Button */}
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          className="flex items-center gap-2 px-5 py-2.5 bg-accent hover:bg-accent-hover disabled:opacity-50 text-white font-medium rounded-xl transition-all duration-200 hover:shadow-lg hover:shadow-accent/20 cursor-pointer disabled:cursor-not-allowed"
        >
          <span className={isRefreshing ? 'animate-spin' : ''}>⟳</span>
          {isRefreshing ? 'Refreshing...' : 'Refresh Model Data'}
        </button>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-5">
          <StatCard
            label="Total Records"
            value={stats.total_records.toLocaleString()}
            color="text-accent"
          />
          <StatCard
            label="Date Range"
            value={`${stats.date_range.start} → ${stats.date_range.end}`}
            color="text-text-primary"
            small
          />
          {Object.entries(stats.commodities).map(([name, data]) => (
            <StatCard
              key={name}
              label={name}
              value={`₱${data.avg_price.toFixed(2)} avg`}
              subValue={`${data.count} records`}
              color={getCommodityColorClass(name)}
            />
          ))}
        </div>
      )}
    </header>
  );
}

function StatCard({
  label,
  value,
  subValue,
  color,
  small,
}: {
  label: string;
  value: string;
  subValue?: string;
  color: string;
  small?: boolean;
}) {
  return (
    <div className="bg-surface/60 rounded-xl p-3 border border-border">
      <p className="text-text-muted text-xs font-medium uppercase tracking-wider">{label}</p>
      <p className={`${color} font-semibold mt-1 ${small ? 'text-xs' : 'text-sm'}`}>{value}</p>
      {subValue && <p className="text-text-muted text-xs mt-0.5">{subValue}</p>}
    </div>
  );
}

function getCommodityColorClass(name: string): string {
  const map: Record<string, string> = {
    Rice: 'text-rice',
    Tomato: 'text-tomato',
    Cabbage: 'text-cabbage',
    Kamote: 'text-kamote',
  };
  return map[name] || 'text-text-primary';
}
