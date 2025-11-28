// components/ActionsPane.tsx
export default function ActionsPane() {
return (
<div className="mt-4 p-4 border rounded-lg bg-card shadow-sm flex items-center justify-between">
<button className="px-4 py-2 rounded-md bg-primary text-primary-foreground">Generate</button>
<div className="flex gap-2">
<button className="p-2 rounded-md bg-muted hover:bg-muted/70">Save</button>
<button className="p-2 rounded-md bg-muted hover:bg-muted/70">Open Folder</button>
<button className="p-2 rounded-md bg-muted hover:bg-muted/70">Clear</button>
</div>
</div>
);
}