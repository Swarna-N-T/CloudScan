// FilePreview.jsx
import React from "react";

export function FilePreviewSuccess({ name, size }) {
  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-green-500/10 border border-green-500/30">
      <div className="flex-shrink-0 size-12 flex items-center justify-center rounded-lg bg-green-500/20">
        <span className="material-symbols-outlined text-3xl">description</span>
      </div>
      <div className="flex-grow">
        <p className="font-medium text-gray-800">{name}</p>
        <p className="text-sm text-gray-600">{(size / 1024).toFixed(1)} KB</p>
      </div>
      <div className="flex-shrink-0 text-green-600">
        <span className="material-symbols-outlined">check_circle</span>
      </div>
    </div>
  );
}

export function FilePreviewError({ name, reason }) {
  return (
    <div className="flex items-center gap-4 p-3 rounded-lg bg-red-500/10 border border-red-500/30">
      <div className="flex-shrink-0 size-12 flex items-center justify-center rounded-lg bg-red-500/20">
        <span className="material-symbols-outlined text-3xl">description</span>
      </div>
      <div className="flex-grow">
        <p className="font-medium text-gray-800">{name}</p>
        <p className="text-sm text-red-600">{reason}</p>
      </div>
      <div className="flex-shrink-0 text-red-600">
        <span className="material-symbols-outlined">cancel</span>
      </div>
    </div>
  );
}
