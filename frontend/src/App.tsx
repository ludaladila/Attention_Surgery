import { useState } from 'react';
import { Brain, Terminal, Download } from 'lucide-react';
import { Header } from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { HeadGrid } from '@/components/surgery/HeadGrid';
import { TokenVisualizer } from '@/components/analysis/TokenVisualizer';
import { LogitLens } from '@/components/analysis/LogitLens';
import { MetricsCard } from '@/components/analysis/MetricsCard';
import { LayerImpactChart } from '@/components/analysis/LayerImpactChart';
import { Tabs } from '@/components/layout/Tabs';
import { AblationMethod, ImportanceMethod, TokenData, MaskGrid, ViewMode, Tab, SurgeryResponse, MetricsData, LayerImpactData } from '@/types';

function App() {
  // State
  const [prompt, setPrompt] = useState("The Eiffel Tower is located in the");
  const [method, setMethod] = useState<AblationMethod>("zero");
  const [importanceMethod, setImportanceMethod] = useState<ImportanceMethod>("gradient");
  const [temperature, setTemperature] = useState(1.0);
  
  // UI State
  const [viewMode, setViewMode] = useState<ViewMode>('structure');
  const [activeTab, setActiveTab] = useState<Tab>('room');
  
  // 12x12 Mask: false = Active, true = Ablated
  const [ablationMask, setAblationMask] = useState<MaskGrid>(
    Array(12).fill(null).map(() => Array(12).fill(false))
  );
  
  const [resultTokens, setResultTokens] = useState<TokenData[]>([]);
  const [controlTokens, setControlTokens] = useState<TokenData[]>([]);
  const [importanceScores, setImportanceScores] = useState<number[][]>([]);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [layerImpacts, setLayerImpacts] = useState<LayerImpactData[]>([]);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Actions
  const toggleHead = (l: number, h: number) => {
    const newMask = [...ablationMask];
    newMask[l] = [...newMask[l]];
    newMask[l][h] = !newMask[l][h];
    setAblationMask(newMask);
  };

  const resetMask = () => {
    setAblationMask(Array(12).fill(null).map(() => Array(12).fill(false)));
  };

  const suggestTopK = () => {
    if (!importanceScores || importanceScores.length === 0) return;

    // Flatten and sort: (score, l, h)
    const flatScores = [];
    for(let l=0; l<12; l++) {
        for(let h=0; h<12; h++) {
            flatScores.push({ score: importanceScores[l][h], l, h });
        }
    }
    flatScores.sort((a, b) => b.score - a.score);
    
    const top5 = flatScores.slice(0, 5);
    const fullCopy = ablationMask.map(row => [...row]);

    top5.forEach(({ l, h }) => {
        fullCopy[l][h] = true; // Ablate it
    });
    setAblationMask(fullCopy);
    setViewMode('structure'); 
  };

  // Helper: Format text to remove GPT-2 artifacts
  const formatText = (text: string) => {
    return text.replace(/Ġ/g, ' ').replace(/Ċ/g, '\n');
  };

  const exportReport = () => {
    const timestamp = new Date().toISOString();
    const ablatedHeads = [];
    for(let l=0; l<12; l++) {
      for(let h=0; h<12; h++) {
        if(ablationMask[l][h]) ablatedHeads.push(`L${l}H${h}`);
      }
    }

    const report = `
# Attention Surgery Report
Date: ${timestamp}

## Configuration
- Prompt: "${prompt}"
- Ablation Method: ${method}
- Importance Method: ${importanceMethod}
- Temperature: ${temperature}

## Ablation State
- Ablated Heads Count: ${ablatedHeads.length}
- Heads: ${ablatedHeads.join(', ') || 'None'}

## Metrics
- KL Divergence: ${metrics?.kl_div?.toFixed(4) || 'N/A'}
- Top-1 Changed: ${(metrics?.top1_changed_ratio || 0) * 100}%
- Perplexity Delta: ${metrics?.perplexity_delta?.toFixed(4) || 'N/A'}
- L2 Diff: ${metrics?.l2_diff?.toFixed(4) || 'N/A'}

## Generated Text (Ablated)
${resultTokens.map(t => formatText(t.text)).join('')}
    `.trim();

    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `surgery-report-${timestamp}.md`;
    a.click();
  };

  const runSurgery = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/surgery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          ablation_mask: ablationMask,
          method,
          importance_method: importanceMethod,
          temperature,
          // max_new_tokens: Use backend default (150)
        })
      });

      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
      }

      const data: SurgeryResponse = await response.json();
      
      setResultTokens(data.generation.ablated_tokens);
      setControlTokens(data.generation.control_tokens);
      setMetrics(data.metrics);
      setImportanceScores(data.importance_scores.gradient_matrix);
      setLayerImpacts(data.layer_impact);

      if (data.generation.ablated_tokens.length > 0) {
        setSelectedTokenIdx(data.generation.ablated_tokens.length - 1);
      }
      
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Computed
  const ablatedCount = ablationMask.flat().filter(Boolean).length;
  const selectedTokenData = selectedTokenIdx !== null ? resultTokens[selectedTokenIdx] : null;

  return (
    <div className="min-h-screen flex text-slate-200 selection:bg-purple-500 selection:text-white bg-slate-900 font-sans">
      
      <Sidebar 
        prompt={prompt}
        setPrompt={setPrompt}
        method={method}
        setMethod={setMethod}
        importanceMethod={importanceMethod}
        setImportanceMethod={setImportanceMethod}
        temperature={temperature}
        setTemperature={setTemperature}
        ablatedCount={ablatedCount}
        resetMask={resetMask}
        onSuggestTopK={suggestTopK}
        onExportReport={exportReport}
      />

      <div className="flex-1 flex flex-col h-screen overflow-hidden">
        
        <Header loading={loading} onRun={runSurgery} />

        <div className="flex-1 overflow-y-auto p-8 space-y-8">
          
          <div className="grid grid-cols-1 xl:grid-cols-[480px_1fr] gap-8 h-full">
            {/* Left: Grid & Controls */}
            <div className="min-w-[480px] flex flex-col gap-6">
              <HeadGrid 
                mask={ablationMask} 
                importanceScores={importanceScores} 
                viewMode={viewMode}
                setViewMode={setViewMode}
                onToggle={toggleHead} 
              />
              
              <MetricsCard metrics={metrics} />
            </div>

            {/* Right: Tabs & Content */}
            <div className="flex flex-col min-w-0 bg-slate-900/30 rounded-2xl border border-slate-800/50 p-1">
              <Tabs activeTab={activeTab} onChange={setActiveTab} />
              
              <div className="flex-1 overflow-y-auto p-4">
                {activeTab === 'room' && (
                   <div className="space-y-6 animate-in fade-in duration-300">
                      <div className="flex items-center justify-between">
                        <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-200">
                          <Terminal className="text-slate-500" /> Token Stream (Ablated)
                        </h2>
                      </div>
                      {error && (
                        <div className="bg-rose-500/20 text-rose-400 p-4 rounded-lg text-sm border border-rose-500/30">
                          Error: {error}
                        </div>
                      )}
                      {resultTokens.length > 0 ? (
                        <TokenVisualizer 
                          tokens={resultTokens} 
                          onTokenClick={setSelectedTokenIdx}
                          selectedTokenIdx={selectedTokenIdx}
                        />
                      ) : (
                        <div className="h-48 border-2 border-dashed border-slate-800 rounded-xl flex flex-col items-center justify-center text-slate-600 gap-4">
                          <Brain size={48} className="opacity-20" />
                          <p>Click "Run Inference" to start the surgery</p>
                        </div>
                      )}
                      
                      {/* Show Control Stream comparison if exists */}
                      {controlTokens.length > 0 && (
                          <div className="mt-8 pt-8 border-t border-slate-800">
                             <h3 className="text-sm font-semibold text-slate-500 mb-4">Control Generation (No Ablation)</h3>
                             <div className="p-4 bg-slate-950/50 rounded-lg text-sm text-slate-400 font-mono leading-relaxed whitespace-pre-wrap">
                                {controlTokens.map(t => formatText(t.text)).join('')}
                             </div>
                          </div>
                      )}
                   </div>
                )}

                {activeTab === 'impact' && (
                  <div className="space-y-6 animate-in fade-in duration-300">
                    <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-200">
                       Local Impact Analysis
                    </h2>
                    <div className="grid grid-cols-1 gap-6">
                        {selectedTokenData && !selectedTokenData.isPrompt ? (
                        <LogitLens tokenData={selectedTokenData} />
                        ) : (
                        <div className="p-8 text-center text-slate-500 bg-slate-800/30 rounded-xl border border-slate-800">
                            Select a generated token to view Logit Lens analysis
                        </div>
                        )}
                        
                        {/* Layer Impact Chart */}
                        {layerImpacts.length > 0 && (
                            <LayerImpactChart data={layerImpacts} />
                        )}
                    </div>
                  </div>
                )}

                {activeTab === 'metrics' && (
                  <div className="space-y-6 animate-in fade-in duration-300">
                     <h2 className="text-lg font-semibold text-slate-200">Detailed Metrics</h2>
                     <MetricsCard metrics={metrics} />
                     {/* Add more detailed metrics visualizations here if needed */}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
