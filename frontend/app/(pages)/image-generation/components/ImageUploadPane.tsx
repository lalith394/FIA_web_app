"use client";

import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import SingleImageUpload from "./SingleImageUpload";
import FolderImageUpload from "./FolderImageUpload";

export default function ImageUploadTabs() {
  return (
    <div className="p-4 rounded-lg border bg-card shadow-sm space-y-4">
      <h3 className="font-semibold text-lg">Upload Images</h3>

      <Tabs defaultValue="single" className="w-full">
        <TabsList className="grid grid-cols-2 w-full">
          <TabsTrigger value="single">Single Image</TabsTrigger>
          <TabsTrigger value="folder">Folder</TabsTrigger>
        </TabsList>

        <TabsContent value="single">
          <SingleImageUpload />
        </TabsContent>

        <TabsContent value="folder">
          <FolderImageUpload />
        </TabsContent>
      </Tabs>
    </div>
  );
}
