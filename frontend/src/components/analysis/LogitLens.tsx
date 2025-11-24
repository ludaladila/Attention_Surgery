import { TokenData } from '@/types';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface LogitLensProps {
  tokenData: TokenData;
}

export const LogitLens = ({ tokenData }: LogitLensProps) => {
  return (
    <div className="bg-slate-800/30 p-6 rounded-xl border border-slate-800 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <h3 className="text-sm font-semibold mb-4 text-slate-400 flex justify-between">
        <span>Logit Lens / Softmax Distribution</span>
        <span className="text-purple-400">Selected: "{tokenData.text}"</span>
      </h3>
      
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={tokenData.topK} layout="vertical" margin={{ left: 40 }}>
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis dataKey="token" type="category" stroke="#94a3b8" fontSize={12} width={60} />
            <Tooltip 
              cursor={{fill: 'rgba(255,255,255,0.05)'}}
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc' }}
            />
            <Bar dataKey="prob" radius={[0, 4, 4, 0]}>
              {tokenData.topK.map((_, index) => (
                <Cell key={`cell-${index}`} fill={index === 0 ? '#a855f7' : '#475569'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

