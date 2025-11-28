"use client";

import { useEffect, useState } from "react";
import { useImageGen } from "../ImageGenContext";

interface Model {
  name: string;
  displayName?: string;
}

export function ModelSelector() {
  const [models, setModels] = useState<Record<string, Model[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await fetch("/api/models");
        const data = (await res.json()) as Record<string, Model[]>;
        setModels(data);
      } catch (error) {
        console.error("Failed to load models:", error);
      }
      setLoading(false);
    };

    loadModels();
  }, []);

  const { model, setModel } = useImageGen();

  return (
    <div className="p-4 rounded-lg border bg-card shadow-sm space-y-3">
      <h3 className="font-medium text-lg">Model Selector</h3>

      <select value={model ?? ""} onChange={(e) => setModel(e.target.value ?? null)} className="w-full p-2 rounded-md bg-muted text-sm border">
        <option>Select model...</option>

        {loading && <option disabled>Loading...</option>}

        {!loading &&
          Object.keys(models).map((category) => (
            <optgroup key={category} label={category[0].toUpperCase() + category.substring(1,category.length).toLowerCase()}>
              {models[category].map((model: Model) => (
                <option key={model.name} value={model.name}>
                  {model.displayName ?? model.name}
                </option>
              ))}
            </optgroup>
          ))}
      </select>
    </div>
  );
}

export default ModelSelector;
