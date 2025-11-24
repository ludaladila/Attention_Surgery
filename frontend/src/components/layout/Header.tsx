import { Layers, ChevronRight, Play, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';

interface HeaderProps {
  loading: boolean;
  onRun: () => void;
}

export const Header = ({ loading, onRun }: HeaderProps) => {
  return (
    <div className="h-16 border-b border-slate-800 bg-slate-900/50 backdrop-blur flex items-center px-8 justify-between sticky top-0 z-50">
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <span className="flex items-center gap-1"><Layers size={14}/> Model: <strong className="text-slate-200">GPT-2 Small</strong></span>
        <ChevronRight size={14} />
        <span className="bg-slate-800 px-2 py-0.5 rounded text-xs border border-slate-700">12 Layers</span>
      </div>
      <button 
        onClick={onRun}
        disabled={loading}
        className={cn(
          "flex items-center gap-2 px-6 py-2 rounded-full font-semibold transition-all shadow-[0_0_20px_rgba(168,85,247,0.3)]",
          loading 
            ? "bg-slate-700 text-slate-500 cursor-not-allowed" 
            : "bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-500 hover:to-pink-500 text-white transform hover:scale-105 active:scale-95"
        )}
      >
        {loading ? <RefreshCw className="animate-spin" size={18}/> : <Play size={18} fill="currentColor" />}
        {loading ? 'Performing Surgery...' : 'Run Inference'}
      </button>
    </div>
  );
};

