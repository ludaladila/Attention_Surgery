import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from 'recharts';
import { LayerImpactData } from '@/types';

interface LayerImpactChartProps {
  data: LayerImpactData[];
}

export const LayerImpactChart = ({ data }: LayerImpactChartProps) => {
  if (!data || data.length === 0) return null;

  return (
    <div className="bg-slate-800/30 p-6 rounded-xl border border-slate-800 h-64 w-full">
        <h3 className="text-sm font-semibold mb-4 text-slate-400">Layer Prediction Confidence (Logit Lens)</h3>
        <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="layer_name" stroke="#94a3b8" fontSize={12} />
                <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={12} />
                <Tooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
                />
                <Line type="monotone" dataKey="impact_score" stroke="#a855f7" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
        </ResponsiveContainer>
    </div>
  );
};

