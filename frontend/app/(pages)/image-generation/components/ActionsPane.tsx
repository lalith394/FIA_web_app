// components/ActionsPane.tsx
"use client";

import { useState } from "react";
import { useImageGen } from "../ImageGenContext";
import { useToast } from "../../../../components/ui/toast";

export default function ActionsPane() {
  const { images, singleImage, model, outputDir, config, generatedOutputs } = useImageGen();
  // local UI state is kept in the ImageGenContext so other panes can read it
  const { setGeneratedOutputs, setLoading, setProgressPercent, reset } = useImageGen();
  const [localLoading, setLocalLoading] = useState(false); // ephemeral
  const toast = useToast();

  const handleGenerate = async () => {
    // reset previous outputs and start progress
    setGeneratedOutputs([]);
    setProgressPercent(5);
    setLoading(true);
    setLocalLoading(true);

    try {
      const form = new FormData();

      // add single image, then folder images
      if (singleImage?.file) {
        form.append("images", singleImage.file, singleImage.file.name);
      }

      for (const entry of images) {
        // if a relativePath exists (directory upload) use it as the filename to preserve folder structure
        const filename = entry.relativePath ?? entry.file.name;
        form.append("images", entry.file, filename);
      }

      // add other form fields
      if (model) form.append("model", model);
      if (outputDir) form.append("output_dir", outputDir);
      form.append("config", JSON.stringify(config));

      // send to backend — using the Flask backend that runs on port 5000
      const res = await fetch("http://localhost:5000/api/generate", {
        method: "POST",
        body: form,
      });
      setProgressPercent(35);
      const data = await res.json();
      if (!res.ok) throw new Error(data?.message ?? "Failed to generate images");

      // Map returned generated URLs into context so other panes can display
      if (Array.isArray(data.generated)) {
        // expected to be full URLs
        setGeneratedOutputs(data.generated);
        setProgressPercent(90);
        toast.show({ title: "Generation complete", description: `${data.generated.length} file(s) generated`, variant: "success", duration: 4000 });
      } else {
        toast.show({ title: "No outputs", description: "No generated files were returned by the server", variant: "info", duration: 4000 });
      }
      // optionally reset state
      // reset();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.show({ title: "Generation failed", description: msg, variant: "error", duration: 8000 });
    } finally {
      setLocalLoading(false);
      // complete progress and clear loading after a short delay so preview updates
      setProgressPercent(100);
      setTimeout(() => {
        setProgressPercent(0);
        setLoading(false);
      }, 700);
    }
  };

  const handleSave = async () => {
    if (!generatedOutputs || generatedOutputs.length === 0) {
      toast.show({ title: 'Nothing to save', description: 'No generated outputs to save yet', variant: 'info', duration: 3000 });
      return;
    }

    try {
      setProgressPercent(2);
      setLoading(true);
      setLocalLoading(true);

      const res = await fetch('http://localhost:5000/api/save_outputs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ urls: generatedOutputs, dest_dir: outputDir }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data?.message ?? 'Unable to save outputs');

      if (Array.isArray(data.saved) && data.saved.length > 0) {
        setGeneratedOutputs(data.saved);
        toast.show({ title: 'Saved outputs', description: `${data.saved.length} file(s) saved to ${outputDir}`, variant: 'success', duration: 4000 });
      } else {
        toast.show({ title: 'Saved', description: 'No files were copied', variant: 'info', duration: 3000 });
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      toast.show({ title: 'Save failed', description: msg, variant: 'error', duration: 7000 });
    } finally {
      setLocalLoading(false);
      setProgressPercent(100);
      setTimeout(() => {
        setProgressPercent(0);
        setLoading(false);
      }, 700);
    }
  };

  const handleClear = () => {
    // Reset global state and generated outputs
    reset();
    toast.show({ title: 'Cleared', description: 'Cleared all selections and outputs', variant: 'info', duration: 2500 });
  };

  return (
    <div className="mt-4 p-4 border rounded-lg bg-card shadow-sm flex items-center justify-between">
      <button onClick={handleGenerate} disabled={localLoading} className="px-4 py-2 rounded-md bg-primary text-primary-foreground">
        {localLoading ? "Generating…" : "Generate"}
      </button>
      <div className="flex gap-2">
        <button onClick={handleSave} disabled={localLoading || !generatedOutputs || generatedOutputs.length===0} className="p-2 rounded-md bg-muted hover:bg-muted/70">Save</button>
        <button
          onClick={() => {
            const rel = (outputDir || '').replace(/^[\\/]+/, '');
            if (rel) window.open(`http://localhost:5000/output/${rel}`, '_blank');
            else window.open('http://localhost:5000/output', '_blank');
          }}
          className="p-2 rounded-md bg-muted hover:bg-muted/70"
        >
          Open Folder
        </button>
        <button onClick={handleClear} className="p-2 rounded-md bg-muted hover:bg-muted/70">Clear</button>
      </div>
      {/* message area replaced by toast — keep small status placeholder for accessibility */}
      {localLoading && <div className="w-full mt-2 text-sm text-muted-foreground">Generating — progress visible in preview pane</div>}
    </div>
);
}