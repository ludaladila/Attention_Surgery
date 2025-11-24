import { Brain, Eye, Grid3X3 } from 'lucide-react';
import { MaskGrid, ViewMode } from '@/types';
import { cn } from '@/lib/utils';
import { useMemo } from 'react';

interface HeadGridProps {
  mask: MaskGrid;
  importanceScores?: number[][]; // 12x12 Matrix
  viewMode: ViewMode;
  setViewMode: (m: ViewMode) => void;
  onToggle: (layer: number, head: number) => void;
}

export const HeadGrid = ({ mask, importanceScores = [], viewMode, setViewMode, onToggle }: HeadGridProps) => {
  const layers = 12;
  const heads = 12;

  // Calculate max score for normalization
  const maxScore = useMemo(() => {
    if (!importanceScores || importanceScores.length === 0) return 0;
    let max = 0;
    for(const row of importanceScores) {
        if(row) max = Math.max(max, ...row);
    }
    return max;
  }, [importanceScores]);

  const getCellColor = (l: number, h: number, isAblated: boolean) => {
    if (viewMode === 'structure') {
      return isAblated 
        ? "bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]" 
        : "bg-emerald-500/20 hover:bg-emerald-500/40 border border-emerald-500/30";
    } else {
      // Importance Mode
      const score = importanceScores[l]?.[h] || 0;
      const normalized = maxScore > 0 ? score / maxScore : 0;
      
      if (isAblated) return "bg-slate-800 border-2 border-rose-500 opacity-50";
      
      if (normalized < 0.1) return "bg-slate-800/50 border border-slate-700";
      if (normalized < 0.3) return "bg-yellow-900/30 border border-yellow-700/50";
      if (normalized < 0.6) return "bg-yellow-600/40 border border-yellow-500/50 shadow-[0_0_5px_rgba(234,179,8,0.2)]";
      return "bg-orange-500/60 border border-orange-400 shadow-[0_0_10px_rgba(249,115,22,0.4)]";
    }
  };

  return (
    <div className="bg-slate-900 p-4 rounded-xl border border-slate-700 shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <Brain size={16} className="text-purple-400" /> 
          Attention Heads
        </h3>
        
        {/* View Mode Toggles */}
        <div className="flex bg-slate-800 rounded-lg p-0.5 border border-slate-700">
          <button
            onClick={() => setViewMode('structure')}
            className={cn(
              "p-1.5 rounded transition-all",
              viewMode === 'structure' ? "bg-slate-700 text-emerald-400 shadow-sm" : "text-slate-500 hover:text-slate-300"
            )}
            title="Structure View"
          >
            <Grid3X3 size={14} />
          </button>
          <button
            onClick={() => setViewMode('importance')}
            className={cn(
              "p-1.5 rounded transition-all",
              viewMode === 'importance' ? "bg-slate-700 text-orange-400 shadow-sm" : "text-slate-500 hover:text-slate-300"
            )}
            title="Importance Heatmap"
          >
            <Eye size={14} />
          </button>
        </div>
      </div>
      
      <div className="grid grid-cols-[auto_1fr] gap-2">
        {/* Y-Axis Labels */}
        <div className="flex flex-col justify-between text-[10px] text-slate-500 pr-2 py-1">
          {Array.from({length: layers}).map((_, i) => <span key={i}>L{i}</span>)}
        </div>
        
        {/* Grid */}
        <div className="grid grid-cols-12 gap-1 aspect-square">
          {Array.from({length: layers}).map((_, l) => 
            Array.from({length: heads}).map((_, h) => {
              const isAblated = mask[l][h];
              const score = importanceScores[l]?.[h] || 0;
              
              return (
                <div 
                  key={`${l}-${h}`}
                  onClick={() => onToggle(l, h)}
                  className={cn(
                    "w-full h-full rounded-[2px] cursor-pointer transition-all duration-200 ease-in-out hover:scale-110 hover:z-10 relative",
                    getCellColor(l, h, isAblated)
                  )}
                  title={`Layer ${l}, Head ${h} | Score: ${score.toFixed(4)}`}
                >
                  {viewMode === 'importance' && score > maxScore * 0.5 && !isAblated && (
                     <div className="absolute inset-0 flex items-center justify-center">
                       <div className="w-1 h-1 bg-white/50 rounded-full animate-pulse" />
                     </div>
                  )}
                </div>
              );
            })
          )}
        </div>
        
        {/* X-Axis Labels */}
        <div></div> {/* Spacer */}
        <div className="flex justify-between text-[10px] text-slate-500 px-1">
          {Array.from({length: heads}).filter((_,i)=>i%2==0).map((_, i) => <span key={i}>H{i*2}</span>)}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 flex gap-4 text-[10px] text-slate-500 justify-center border-t border-slate-800 pt-3">
        {viewMode === 'structure' ? (
          <>
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-emerald-500/40 border border-emerald-500/50 rounded-sm"></div> Active</div>
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-rose-500 rounded-sm"></div> Ablated</div>
          </>
        ) : (
          <>
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-800 border border-slate-700 rounded-sm"></div> Low</div>
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-yellow-900/50 border border-yellow-700 rounded-sm"></div> Med</div>
            <div className="flex items-center gap-1"><div className="w-2 h-2 bg-orange-500 border border-orange-400 rounded-sm"></div> High</div>
             <div className="flex items-center gap-1"><div className="w-2 h-2 bg-slate-800 border border-rose-500 rounded-sm"></div> Ablated</div>
          </>
        )}
      </div>
    </div>
  );
};
