"use client";

import { useState } from "react";
import { useImageGen } from "../ImageGenContext";
import Image from "next/image";

export default function FolderImageUpload() {
  const { images: ctxImages, setImages: setCtxImages } = useImageGen();
  const [folderPath, setFolderPath] = useState<string | null>(null);

  // Local previews (limited to first N) — derived from context.images for display
  const images = ctxImages.slice(0, 20).map((i) => i.preview ?? "");

  const handleFolder = (files: FileList) => {
    const arr = [] as { file: File; relativePath?: string; preview?: string }[];
    let baseFolder = "";

    Array.from(files).forEach((file, index) => {
      if (index === 0) {
        baseFolder = file.webkitRelativePath.split("/")[0];
        setFolderPath(baseFolder);
      }

      if (file.type.startsWith("image/")) {
        const url = URL.createObjectURL(file);
        const wp = (file as unknown as { webkitRelativePath?: string }).webkitRelativePath;
        arr.push({ file, relativePath: wp, preview: url });
      }
    });

    // store all images into context (we will limit previews in UI)
    setCtxImages(arr);
  };

  return (
    <div className="space-y-3">
      {/* clickable drop area */}
      <div
        onClick={() => document.getElementById("folderInput")?.click()}
        className="w-full h-40 border rounded-lg bg-muted flex items-center justify-center cursor-pointer"
      >
        <p className="text-muted-foreground text-sm">
          Click to choose a folder containing images
        </p>
      </div>

      {/* hidden directory input */}
      <input
        id="folderInput"
        type="file"
        className="hidden"
        multiple
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        //@ts-expect-error
        webkitdirectory="true"
        onChange={(e) => handleFolder(e.target.files!)}
      />

      {/* directory path */}
      {folderPath && (
        <p className="text-sm text-muted-foreground">
          Folder: {folderPath}
        </p>
      )}

      {/* preview gallery */}
      {images.length > 0 && (
        <div className="max-h-56 overflow-y-auto grid grid-cols-3 gap-2 p-2 border rounded-lg">
          {images.map((src, idx) => (
            <Image
              key={idx}
              src={src}
              alt="Mini preview"
              width={120}
              height={120}
              className="object-cover rounded-md"
            />
          ))}
        </div>
      )}
    </div>
  );
}
