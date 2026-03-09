export default function FileUpload() {
  return (
    <div className="relative">
      <input
        accept=".json,.yaml,.yml"
        className="hidden"
        id="file-upload"
        type="file"
      />
      <label
        htmlFor="file-upload"
        id="dropzone"
        className="w-full flex flex-col items-center justify-center gap-2 px-4 py-10 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-600 text-gray-500 dark:text-gray-400 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
      >
        <span className="material-symbols-outlined text-4xl">cloud_upload</span>
        <span className="font-medium">Drag &amp; drop your file here</span>
        <span className="text-xs">or</span>
        <span className="bg-white dark:bg-gray-700/50 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm font-medium">
          Choose File
        </span>
      </label>
    </div>
  );
}
