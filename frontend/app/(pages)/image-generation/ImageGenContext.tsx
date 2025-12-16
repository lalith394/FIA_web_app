"use client";

import React, { createContext, useContext, useState } from "react";

export interface ImageFile {
  file: File;
  relativePath?: string;
  preview?: string;
}

type Config = {
  threshold: number;
  batchSize: number;
  saveFeatures?: boolean;
};

export interface ImageGenState {
  images: ImageFile[]; // folder uploads
  setImages: (v: ImageFile[]) => void;
  singleImage: ImageFile | null; // single upload
  setSingleImage: (v: ImageFile | null) => void;
  model: string | null;
  setModel: (v: string | null) => void;
  outputDir: string;
  setOutputDir: (v: string) => void;
  config: Config;
  setConfig: (patch: Partial<Config>) => void;
  // outputs produced by the backend as accessible URLs
  generatedOutputs: string[];
  setGeneratedOutputs: (v: string[]) => void;
  // loading / progress helpers to show progress in UI
  loading: boolean;
  setLoading: (v: boolean) => void;
  progressPercent: number;
  setProgressPercent: (v: number) => void;
  reset: () => void;
}

const defaultState: ImageGenState = {
  images: [],
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setImages: () => {},
  singleImage: null,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setSingleImage: () => {},
  model: null,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setModel: () => {},
  outputDir: `/outputs/${new Date().toISOString().slice(0, 10)}/`,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setOutputDir: () => {},
  config: { threshold: 0.5, batchSize: 1, saveFeatures: false },
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setConfig: () => {},
  generatedOutputs: [],
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setGeneratedOutputs: () => {},
  loading: false,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setLoading: () => {},
  progressPercent: 0,
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  setProgressPercent: () => {},
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  reset: () => {},
};

const ImageGenContext = createContext<ImageGenState>(defaultState);

export const ImageGenProvider: React.FC<React.PropsWithChildren<object>> = ({ children }) => {
  const [images, setImages] = useState<ImageFile[]>([]);
  const [singleImage, setSingleImage] = useState<ImageFile | null>(null);
  const [model, setModel] = useState<string | null>(null);
  const [outputDir, setOutputDir] = useState<string>(defaultState.outputDir);
  const [config, rawSetConfig] = useState<Config>(defaultState.config);
  const [generatedOutputs, setGeneratedOutputs] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [progressPercent, setProgressPercent] = useState<number>(0);

  const setConfig = (patch: Partial<Config>) => rawSetConfig((cur) => ({ ...cur, ...patch }));

  const reset = () => {
    setImages([]);
    setSingleImage(null);
    setModel(null);
    setOutputDir(`/outputs/${new Date().toISOString().slice(0, 10)}/`);
    rawSetConfig(defaultState.config);
    // clear generated outputs / progress
    setGeneratedOutputs([]);
    setLoading(false);
    setProgressPercent(0);
  };

  return (
    <ImageGenContext.Provider value={{ images, setImages, singleImage, setSingleImage, model, setModel, outputDir, setOutputDir, config, setConfig, generatedOutputs, setGeneratedOutputs, loading, setLoading, progressPercent, setProgressPercent, reset }}>
      {children}
    </ImageGenContext.Provider>
  );
};

export const useImageGen = () => useContext(ImageGenContext);

export default ImageGenContext;
