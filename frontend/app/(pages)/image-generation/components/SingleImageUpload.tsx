"use client";

import { useState } from "react";
import { useImageGen } from "../ImageGenContext";
import Image from "next/image";
import { X, Loader2 } from "lucide-react";

export default function SingleImageUpload() {
  const { singleImage, setSingleImage } = useImageGen();
  const [preview, setPreview] = useState<string | null>(singleImage?.preview ?? null);
  const [fileName, setFileName] = useState<string | null>(singleImage?.file.name ?? null);
  const [loading, setLoading] = useState(false);

  const handleFile = (file: File) => {
    setLoading(true);
    setFileName(file.name);

    const reader = new FileReader();
    reader.onload = () => {
      // set local preview and push to global state
      const previewStr = reader.result as string;
      setPreview(previewStr);
      setSingleImage({ file, preview: previewStr });
      setLoading(false);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="space-y-3">
      <div
        onClick={() => document.getElementById("singleInput")?.click()}
        className="w-full h-64 border rounded-lg bg-muted flex items-center justify-center overflow-hidden cursor-pointer relative"
      >
        {/* remove button */}
        {preview && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setPreview(null);
              setFileName(null);
              setSingleImage(null);
            }}
            className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 p-1 rounded-full"
          >
            <X className="w-5 h-5 text-white" />
          </button>
        )}

        {/* loader */}
        {loading && <Loader2 className="animate-spin text-primary w-8 h-8" />}

        {/* preview */}
        {!loading && preview && (
          <Image
            src={preview}
            alt="Preview"
            width={400}
            height={400}
            className="object-contain rounded-md"
          />
        )}

        {/* default */}
        {!preview && !loading && (
          <p className="text-sm text-muted-foreground">
            Drop image or click to upload
          </p>
        )}
      </div>

      {/* hidden input */}
      <input
        id="singleInput"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      {/* filename */}
      {fileName && (
        <p className="text-sm text-muted-foreground">Selected: {fileName}</p>
      )}
    </div>
  );
}
