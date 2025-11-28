"use client"
import React, { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Upload, FolderOpen, ImageIcon } from "lucide-react";

export default function ImageGenerationPage() {
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const [uploadedFolder, setUploadedFolder] = useState<FileList | null>(null);

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-8">
      <h1 className="text-3xl font-bold tracking-tight">Image Generation</h1>
      <p className="text-muted-foreground text-sm">Upload images and select a model to generate segmentation results.</p>

      <Tabs defaultValue="single" className="w-full">
        <TabsList className="grid w-full grid-cols-2 rounded-xl">
          <TabsTrigger value="single">Single Image</TabsTrigger>
          <TabsTrigger value="multiple">Multiple Images</TabsTrigger>
        </TabsList>

        {/* Single Image Tab */}
        <TabsContent value="single" className="mt-6">
          <Card className="border rounded-xl shadow-sm">
            <CardHeader>
              <CardTitle>Upload a Single Image</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex flex-col space-y-4">
                <label className="text-sm font-medium">Choose Image</label>
                <div className="flex items-center gap-4">
                  <Input
                    type="file"
                    accept="image/*"
                    onChange={(e) => setUploadedImage(e.target.files?.[0] || null)}
                  />
                  <Button variant="outline" className="gap-2">
                    <Upload className="w-4 h-4" /> Upload
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Segmentation Model</label>
                <Select onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="unet">UNet</SelectItem>
                    <SelectItem value="deeplabv3">DeepLabV3</SelectItem>
                    <SelectItem value="attentionunet">Attention UNet</SelectItem>
                    <SelectItem value="smp_unet">SMP · UNet (ResNet34)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button className="w-full" disabled={!uploadedImage || !selectedModel}>
                Generate Segmentation
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Multiple Image Tab */}
        <TabsContent value="multiple" className="mt-6">
          <Card className="border rounded-xl shadow-sm">
            <CardHeader>
              <CardTitle>Upload a Folder of Images</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <label className="text-sm font-medium">Choose Folder</label>
                <Input
                  type="file"
                  webkitdirectory="true"
                  directory="true"
                  onChange={(e) => setUploadedFolder(e.target.files || null)}
                />
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <FolderOpen className="w-4 h-4" /> Folder Input Supported
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">Segmentation Model</label>
                <Select onValueChange={setSelectedModel}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="unet">UNet</SelectItem>
                    <SelectItem value="deeplabv3">DeepLabV3</SelectItem>
                    <SelectItem value="attentionunet">Attention UNet</SelectItem>
                    <SelectItem value="smp_unet">SMP · UNet (ResNet34)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button className="w-full" disabled={!uploadedFolder || !selectedModel}>
                Process Folder
              </Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
