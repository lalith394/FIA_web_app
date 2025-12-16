"use client";

import OutputDirectorySelector from "./OutputDirectorySelector";

export default function OutputDirectoryPane() {
  return (
    <div className="p-4 rounded-lg border bg-card shadow-sm space-y-4">
      <h3 className="font-semibold text-lg">Output Directory</h3>
      <OutputDirectorySelector />
    </div>
  );
}