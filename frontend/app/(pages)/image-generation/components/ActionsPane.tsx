// components/ActionsPane.tsx
"use client";

import { useState } from "react";
import { useImageGen } from "../ImageGenContext";

export default function ActionsPane() {
  const { images, singleImage, model, outputDir, config } = useImageGen();
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const handleGenerate = async () => {
    setMessage(null);
    setLoading(true);

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

      const data = await res.json();

      if (!res.ok) throw new Error(data?.message ?? "Failed to generate images");

      setMessage(`Success: ${data.message ?? "Generation started"}`);
      // optionally reset state
      // reset();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setMessage(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mt-4 p-4 border rounded-lg bg-card shadow-sm flex items-center justify-between">
      <button onClick={handleGenerate} disabled={loading} className="px-4 py-2 rounded-md bg-primary text-primary-foreground">
        {loading ? "Generating…" : "Generate"}
      </button>
      <div className="flex gap-2">
        <button className="p-2 rounded-md bg-muted hover:bg-muted/70">Save</button>
        <button className="p-2 rounded-md bg-muted hover:bg-muted/70">Open Folder</button>
        <button className="p-2 rounded-md bg-muted hover:bg-muted/70">Clear</button>
      </div>
      {message && <div className="w-full mt-2 text-sm text-muted-foreground">{message}</div>}
    </div>
);
}