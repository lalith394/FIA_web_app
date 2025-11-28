// components/ModelSelector.tsx
export function ModelSelector() {
return (
<div className="p-4 rounded-lg border bg-card shadow-sm space-y-3">
<h3 className="font-medium text-lg">Model Selector</h3>
<select className="w-full p-2 rounded-md bg-muted text-sm border">
<option>Select model...</option>
<option>UNet Retina Segmentation</option>
<option>DR Grade Classifier</option>
<option>OD/OC Cup Segmentation</option>
</select>
</div>
);
}


export default ModelSelector;