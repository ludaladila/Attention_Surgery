import { TokenData } from '@/types';
import { cn } from '@/lib/utils';

interface TokenVisualizerProps {
  tokens: TokenData[];
  selectedTokenIdx: number | null;
  onTokenClick: (idx: number) => void;
}

const formatToken = (text: string) => {
  if (!text) return "";
  // Replace GPT-2 special characters
  return text.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
};

export const TokenVisualizer = ({ tokens, selectedTokenIdx, onTokenClick }: TokenVisualizerProps) => {
  return (
    <div className="flex flex-wrap gap-2 bg-slate-800/50 p-6 rounded-xl border border-slate-700 min-h-[120px]">
      {tokens.map((token, idx) => {
        // Calculate purple opacity based on probability
        const alpha = token.isPrompt ? 0.1 : 0.2 + (token.prob * 0.8);
        const bgColor = token.isPrompt ? 'rgba(148, 163, 184, 0.2)' : `rgba(168, 85, 247, ${alpha})`;
        const isSelected = selectedTokenIdx === idx;
        
        return (
          <div
            key={idx}
            onClick={() => onTokenClick(idx)}
            className={cn(
              "px-3 py-1.5 rounded-md text-sm font-mono cursor-pointer border transition-all duration-200 ease-[cubic-bezier(0.4,0,0.2,1)] hover:-translate-y-0.5 hover:shadow-lg hover:shadow-purple-500/50 whitespace-pre",
              token.isPrompt ? "text-slate-400" : "text-white font-medium",
              isSelected ? "border-pink-400" : "border-transparent"
            )}
            style={{ backgroundColor: bgColor }}
            title={`Prob: ${(token.prob * 100).toFixed(2)}%`}
          >
            {formatToken(token.text)}
            {!token.isPrompt && (
              <div className="text-[9px] opacity-70 mt-1">
                {(token.prob * 100).toFixed(1)}%
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};
