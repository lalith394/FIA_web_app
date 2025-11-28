"use client";

import { useImageGen } from "../ImageGenContext";

export default function ModelConfigPane() {
	const { config, setConfig } = useImageGen();

	return (
<div className="p-4 rounded-lg border bg-card shadow-sm space-y-4">
<h3 className="font-medium text-lg">Model Settings</h3>
<div className="space-y-2">
<label className="text-sm font-medium">Segmentation Threshold</label>
		<input value={config.threshold} onChange={(e) => setConfig({ threshold: Number(e.target.value) })} type="range" min="0" max="1" step="0.01" className="w-full" />
</div>
			<div className="space-y-2">
				<label className="text-sm font-medium">Batch Size</label>
				<input
					value={config.batchSize}
					onChange={(e) => setConfig({ batchSize: Number(e.target.value || 1) })}
					type="number"
					min={1}
					className="border p-2 rounded w-full"
				/>
			</div>
	  <div className="space-y-2">
		<label className="text-sm font-medium">Save features (d4)</label>
		<label className="flex items-center gap-2 text-sm">
		  <input type="checkbox" checked={!!config.saveFeatures} onChange={(e) => setConfig({ saveFeatures: e.target.checked })} />
		  <span className="ml-2">Enabled</span>
		</label>
	  </div>
</div>
);
}