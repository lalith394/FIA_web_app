export default function ModelConfigPane() {
return (
<div className="p-4 rounded-lg border bg-card shadow-sm space-y-4">
<h3 className="font-medium text-lg">Model Settings</h3>
<div className="space-y-2">
<label className="text-sm font-medium">Segmentation Threshold</label>
<input type="range" min="0" max="1" step="0.01" className="w-full" />
</div>
<div className="space-y-2">
<label className="text-sm font-medium">Batch Size</label>
<input type="number" className="border p-2 rounded w-full" defaultValue={1} />
</div>
<div className="space-y-2">
<label className="text-sm font-medium">Preprocessing Mode</label>
<select className="w-full p-2 border rounded bg-muted">
<option>Normalize Only</option>
<option>CLAHE</option>
<option>Vessel Enhancement</option>
</select>
</div>
</div>
);
}