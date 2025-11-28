// components/OutputPreviewPane.tsx
"use client";
 

import { useImageGen } from "../ImageGenContext";
import Image from "next/image";

export default function OutputPreviewPane() {
	const { generatedOutputs, loading, progressPercent } = useImageGen();

	return (
		<div className="flex-1 border rounded-lg bg-card shadow-sm p-4 flex items-center justify-center overflow-auto flex-col gap-4">
			{loading ? (
				<div className="w-full">
					<div className="text-sm text-muted-foreground mb-2">Generating — please wait</div>
					<div className="w-full bg-muted rounded-md h-3 overflow-hidden">
						<div
							className="h-3 bg-primary"
							style={{ width: `${Math.max(5, progressPercent)}%`, transition: 'width 300ms ease' }}
						/>
					</div>
				</div>
			) : generatedOutputs && generatedOutputs.length > 0 ? (
				<div className="w-full h-full grid grid-cols-1 gap-3 justify-center items-start overflow-auto">
					{generatedOutputs.map((url, idx) => (
						<div key={`${url}-${idx}`} className="rounded-md border p-2 bg-white/5">
							<div className="text-xs text-muted-foreground mb-1">Output #{idx + 1}</div>
							  <Image src={url} alt={`generated-${idx + 1}`} width={576} height={384} unoptimized className="max-h-[420px] w-auto mx-auto block" />
						</div>
					))}
				</div>
			) : (
				<div className="text-muted-foreground text-sm">Generated output will appear here</div>
			)}
		</div>
	);
}