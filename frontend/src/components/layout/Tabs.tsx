import { cn } from '@/lib/utils';
import { Tab } from '@/types';

interface TabsProps {
  activeTab: Tab;
  onChange: (tab: Tab) => void;
}

export const Tabs = ({ activeTab, onChange }: TabsProps) => {
  const tabs: { id: Tab; label: string }[] = [
    { id: 'room', label: 'Surgery Room' },
    { id: 'impact', label: 'Impact Analysis' },
    { id: 'metrics', label: 'Metrics Room' },
  ];

  return (
    <div className="flex border-b border-slate-800 mb-6">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "px-6 py-3 text-sm font-medium transition-all relative",
            activeTab === tab.id 
              ? "text-purple-400" 
              : "text-slate-500 hover:text-slate-300"
          )}
        >
          {tab.label}
          {activeTab === tab.id && (
            <div className="absolute bottom-0 left-0 w-full h-0.5 bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.5)]" />
          )}
        </button>
      ))}
    </div>
  );
};

