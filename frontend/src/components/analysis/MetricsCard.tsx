import { Activity, BarChart2, ArrowRightLeft, Scale } from 'lucide-react';
import { MetricsData } from '@/types';

interface MetricsCardProps {
  metrics: MetricsData | null;
}

export const MetricsCard = ({ metrics }: MetricsCardProps) => {
  if (!metrics) return null;

  const klDiv = metrics.kl_div || 0;
  const top1Change = (metrics.top1_changed_ratio || 0) * 100;
  const perplexity = metrics.perplexity_delta || 0;
  const l2Diff = metrics.l2_diff || 0;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-2">
        <div className="text-xs text-slate-500 uppercase flex items-center gap-2">
          <ArrowRightLeft size={14} /> KL Divergence
        </div>
        <div className="text-2xl font-mono text-slate-200">{klDiv.toFixed(4)}</div>
      </div>
      
      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-2">
        <div className="text-xs text-slate-500 uppercase flex items-center gap-2">
          <Activity size={14} /> Top-1 Changed
        </div>
        <div className="text-2xl font-mono text-rose-400">{top1Change.toFixed(1)}%</div>
      </div>

      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-2">
        <div className="text-xs text-slate-500 uppercase flex items-center gap-2">
          <BarChart2 size={14} /> Perplexity Δ
        </div>
        <div className="text-2xl font-mono text-slate-200">{perplexity > 0 ? '+' : ''}{perplexity.toFixed(2)}</div>
      </div>

      <div className="bg-slate-800/50 p-4 rounded-xl border border-slate-700 flex flex-col gap-2">
        <div className="text-xs text-slate-500 uppercase flex items-center gap-2">
          <Scale size={14} /> L2 Diff
        </div>
        <div className="text-2xl font-mono text-slate-200">{l2Diff.toFixed(2)}</div>
      </div>
    </div>
  );
};
