export type AblationMethod = 'zero' | 'mean' | 'random' | 'previous';
export type ImportanceMethod = 'gradient' | 'rollback' | 'ablation';
export type ViewMode = 'structure' | 'importance';
export type Tab = 'room' | 'impact' | 'metrics';

export interface TopKItem {
  token: string;
  prob: number;
}

export interface TokenData {
  id: number;
  text: string;
  prob: number;
  isPrompt: boolean;
  topK: TopKItem[];
}

export type MaskGrid = boolean[][];

export interface SurgeryRequest {
  prompt: string;
  ablation_mask: MaskGrid;
  method: AblationMethod;
  importance_method: ImportanceMethod;
  temperature: number;
  max_new_tokens: number;
}

export interface MetricsData {
  kl_div: number;
  perplexity_delta: number;
  top1_changed_ratio: number;
  l2_diff: number;
}

export interface GenerationData {
  control_tokens: TokenData[];
  ablated_tokens: TokenData[];
}

export interface ImportanceScoresData {
  gradient_matrix: number[][];
}

export interface LayerImpactData {
  layer_name: string;
  impact_score: number;
}

export interface SurgeryResponse {
  metrics: MetricsData;
  generation: GenerationData;
  importance_scores: ImportanceScoresData;
  layer_impact: LayerImpactData[];
}
