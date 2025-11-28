import React from "react";
import ModelSelector from "./components/ModelSelector";
import ImageUploadPane from "./components/ImageUploadPane";
import ModelConfigPane from "./components/ModelConfigPane";
import OutputPreviewPane from "./components/OutputPreviewPane";
import ActionsPane from "./components/ActionsPane";
import OutputDirectoryPane from "./components/OutputDirectoryPane";


export default function ImageGenerationPage() {
return (
<div className="w-full h-[calc(100vh-70px)] flex bg-background text-foreground overflow-hidden">
  {/* LEFT PANEL */}
  <div className="w-[420px] min-w-[420px] border-r h-full p-4 flex flex-col gap-4 overflow-y-auto">
    <h2 className="text-xl font-semibold tracking-tight">Image Generation</h2>
    <ModelSelector />
    <ImageUploadPane />
    <OutputDirectoryPane />
    <ModelConfigPane />
  </div>


  {/* RIGHT PANEL */}
  <div className="flex-1 h-full   p-4 overflow-hidden flex flex-col">
    <OutputPreviewPane />
    <ActionsPane />
  </div>
</div>
);
}