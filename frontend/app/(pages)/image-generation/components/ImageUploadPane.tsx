export default function ImageUploadPane() {
return (
<div className="p-4 rounded-lg border bg-card shadow-sm space-y-4">
<h3 className="font-medium text-lg">Upload Image</h3>
<div className="border rounded-lg h-40 flex items-center justify-center bg-muted/40">
<span className="text-muted-foreground text-sm">Drop image or click to upload</span>
</div>
<input type="file" className="text-sm" />
</div>
);
}