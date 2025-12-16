/* eslint-disable @typescript-eslint/ban-ts-comment */
"use client";

import { useEffect, useState } from "react";
import { useImageGen } from "../ImageGenContext";
import { FolderOpen } from "lucide-react";

export default function OutputDirectorySelector() {
  const { outputDir, setOutputDir } = useImageGen();

  // Keep a local 'display' default but the context holds the real source of truth
  const [localDir] = useState<string>(() => {
    // Generate default directory = outputs/YYYY-MM-DD/
    const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
    return `/outputs/${today}/`;
  });

  const handleFolderPick = (files: FileList | null) => {
    if (!files || files.length === 0) return;

    // Extract base directory of selection
    const path = files[0].webkitRelativePath;
    const rootFolder = path.split("/")[0]; // get first folder

    setOutputDir(rootFolder);
  };

  return (
    <div className="p-4 rounded-lg border bg-card shadow-sm space-y-3">
      <h3 className="font-semibold text-lg">Output Directory</h3>

      <button
        onClick={() => document.getElementById("outputDirInput")?.click()}
        className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-muted hover:bg-muted/80 transition"
      >
        <FolderOpen className="w-5 h-5" />
        Choose Output Folder
      </button>

      {/* hidden folder selector */}
      <input
        id="outputDirInput"
        type="file"
        //@ts-ignore
        webkitdirectory="true"
        directory=""
        multiple
        className="hidden"
        onChange={(e) => handleFolderPick(e.target.files)}
      />

      <p className="text-sm text-muted-foreground break-all">
        <span className="font-medium text-foreground">Selected:</span>{" "}
        {outputDir ?? localDir}
      </p>
    </div>
  );
}
