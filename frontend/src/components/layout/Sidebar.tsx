import { Activity, Settings, RefreshCw, Wand2, FileText } from 'lucide-react';
import { AblationMethod, ImportanceMethod } from '@/types';
import { cn } from '@/lib/utils';

interface SidebarProps {
  prompt: string;
  setPrompt: (v: string) => void;
  method: AblationMethod;
  setMethod: (v: AblationMethod) => void;
  importanceMethod: ImportanceMethod;
  setImportanceMethod: (v: ImportanceMethod) => void;
  temperature: number;
  setTemperature: (v: number) => void;
  ablatedCount: number;
  resetMask: () => void;
  onSuggestTopK: () => void;
  onExportReport: () => void;
}

export const Sidebar = ({
  prompt,
  setPrompt,
  method,
  setMethod,
  importanceMethod,
  setImportanceMethod,
  temperature,
  setTemperature,
  ablatedCount,
  resetMask,
  onSuggestTopK,
  onExportReport
}: SidebarProps) => {
  return (
    <div className="w-80 bg-slate-900 border-r border-slate-800 p-6 flex flex-col gap-8 h-screen sticky top-0 overflow-y-auto shrink-0">
      <div>
        <h1 className="text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent flex items-center gap-2">
          <Activity /> Attention Surgery
        </h1>
        <p className="text-xs text-slate-500 mt-1">Interactive Mechanistic Interpretability</p>
      </div>

      {/* Input Section */}
      <div className="space-y-4">
        <label className="text-xs font-semibold uppercase text-slate-500 tracking-wider">Input Prompt</label>
        <textarea 
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm focus:ring-2 focus:ring-purple-500 focus:outline-none transition-all h-24 resize-none font-mono text-slate-200 placeholder:text-slate-600"
          placeholder="Enter text to analyze..."
        />
      </div>

      {/* Config Section */}
      <div className="space-y-6">
        <label className="text-xs font-semibold uppercase text-slate-500 tracking-wider flex items-center gap-2">
          <Settings size={14} /> Surgery Config
        </label>
        
        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Ablation Method</span>
          </div>
          <div className="grid grid-cols-2 gap-1 bg-slate-800 p-1 rounded-lg">
            {(['zero', 'mean', 'random', 'previous'] as AblationMethod[]).map(m => (
              <button 
                key={m}
                onClick={() => setMethod(m)}
                className={cn(
                  "text-xs py-1.5 rounded capitalize transition-all",
                  method === m ? "bg-purple-600 text-white shadow-md" : "text-slate-400 hover:text-white"
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Importance Method</span>
          </div>
          <div className="grid grid-cols-3 gap-1 bg-slate-800 p-1 rounded-lg">
            {(['gradient', 'rollback', 'ablation'] as ImportanceMethod[]).map(m => (
              <button 
                key={m}
                onClick={() => setImportanceMethod(m)}
                className={cn(
                  "text-xs py-1.5 rounded capitalize transition-all",
                  importanceMethod === m ? "bg-purple-600 text-white shadow-md" : "text-slate-400 hover:text-white"
                )}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Temperature</span>
            <span className="text-purple-400 font-mono">{temperature}</span>
          </div>
          <input 
            type="range" min="0.1" max="2.0" step="0.1" 
            value={temperature}
            onChange={(e) => setTemperature(parseFloat(e.target.value))}
            className="w-full h-1 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
        </div>
      </div>

      {/* Actions */}
      <div className="space-y-3 pt-4 border-t border-slate-800">
        <button
           onClick={onSuggestTopK}
           className="w-full py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-purple-400 border border-purple-500/20 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <Wand2 size={14} /> Ablate Top-5 Heads
        </button>
        <button
           onClick={onExportReport}
           className="w-full py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 rounded-lg transition-colors flex items-center justify-center gap-2"
        >
          <FileText size={14} /> Export Report
        </button>
      </div>

      {/* Stats */}
      <div className="mt-auto bg-slate-800/50 p-4 rounded-lg border border-slate-700">
        <div className="text-xs text-slate-500 mb-2">Surgery Stats</div>
        <div className="flex justify-between items-end">
          <div>
            <div className="text-2xl font-bold text-rose-400">{ablatedCount}</div>
            <div className="text-[10px] text-slate-400">Heads Ablated</div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-emerald-400">{144 - ablatedCount}</div>
            <div className="text-[10px] text-slate-400">Heads Active</div>
          </div>
        </div>
        {ablatedCount > 0 && (
          <button 
            onClick={resetMask}
            className="w-full mt-3 py-1 text-xs text-slate-400 hover:text-white border border-slate-700 hover:bg-slate-700 rounded transition-colors flex justify-center gap-1 items-center"
          >
            <RefreshCw size={10} /> Reset All
          </button>
        )}
      </div>
    </div>
  );
};
