import { useState, useRef } from "react";
import axios from "axios";
import { FilePreviewSuccess, FilePreviewError } from "../components/FilePreview";
import { ToastSuccess, ToastError } from "../components/Toast";
import ResultsDisplay from "../components/ResultsDisplay";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export default function UploadPage() {
  const [files, setFiles] = useState([]);
  const [toast, setToast] = useState(null);
  const [error, setError] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [results, setResults] = useState(null);
  const [enableRemediation, setEnableRemediation] = useState(true);
  const folderInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    if (!selectedFiles.length) return;

    const tfFiles = selectedFiles.filter((f) => f.name.endsWith(".tf"));
    const otherAllowed = selectedFiles.filter(
      (f) => f.name.endsWith(".json") || f.name.endsWith(".tf")
    );

    if (tfFiles.length === 0) {
      setError(true);
      setToast("error");
      setFiles([]);
      return;
    }

    setFiles(otherAllowed);
    setError(false);
    setToast("success");
    setResults(null);
  };

  const handleScan = async () => {
    if (files.length === 0) {
      setToast("error");
      return;
    }

    setScanning(true);
    setResults(null);
    setToast(null);

    try {
      const formData = new FormData();
      files.forEach((file) => {
        // Use webkitRelativePath if available (folder upload), otherwise just filename
        const path = file.webkitRelativePath || file.name;
        formData.append("files", file, path);
      });

      const response = await axios.post(
        `${API_URL}/api/scan?remediation=${enableRemediation}`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 120000, // 2 min timeout for LLM calls
        }
      );

      setResults(response.data);
      setToast("success");
    } catch (err) {
      console.error("Scan failed:", err);
      setError(true);
      setToast("error");
    } finally {
      setScanning(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 flex flex-col">
      {/* 🔹 HEADER */}
      <header className="bg-white shadow-sm px-8 py-5 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-indigo-600">CloudScan</h1>
          <p className="text-sm text-gray-500">
            AI-Powered Cloud Misconfiguration Analyzer
          </p>
        </div>
        <span className="material-symbols-outlined text-indigo-500 text-3xl">
          security
        </span>
      </header>

      {/* 🔹 MAIN CONTENT */}
      <main className="flex-grow px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Upload Card */}
          <div className="bg-white rounded-2xl shadow-xl p-10">
            {/* Title */}
            <h2 className="text-3xl font-semibold text-gray-800 mb-2">
              Upload Configuration
            </h2>
            <p className="text-gray-500 mb-8">
              Upload Terraform files or an entire project folder to scan for
              security misconfigurations.
            </p>

            {/* Upload Options Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
              {/* File Upload */}
              <label
                htmlFor="fileUpload"
                className="flex flex-col items-center justify-center border-2 border-dashed border-indigo-300 rounded-xl p-8 cursor-pointer hover:border-indigo-500 hover:bg-indigo-50 transition-all duration-300"
              >
                <span className="material-symbols-outlined text-5xl text-indigo-400 mb-3">
                  upload_file
                </span>
                <p className="font-medium text-gray-700">Upload Files</p>
                <p className="text-sm text-gray-500 mt-1">Select .tf files</p>
                <input
                  id="fileUpload"
                  type="file"
                  className="hidden"
                  multiple
                  accept=".tf,.json"
                  onChange={handleFileChange}
                />
              </label>

              {/* Folder Upload */}
              <label
                htmlFor="folderUpload"
                className="flex flex-col items-center justify-center border-2 border-dashed border-purple-300 rounded-xl p-8 cursor-pointer hover:border-purple-500 hover:bg-purple-50 transition-all duration-300"
              >
                <span className="material-symbols-outlined text-5xl text-purple-400 mb-3">
                  folder_open
                </span>
                <p className="font-medium text-gray-700">Upload Folder</p>
                <p className="text-sm text-gray-500 mt-1">
                  Entire Terraform project
                </p>
                <input
                  id="folderUpload"
                  ref={folderInputRef}
                  type="file"
                  className="hidden"
                  onChange={handleFileChange}
                  {...{ webkitdirectory: "", directory: "" }}
                />
              </label>
            </div>

            {/* File Preview */}
            {files.length > 0 && !error && (
              <div className="mb-4 bg-indigo-50 rounded-lg p-4">
                <p className="text-sm font-medium text-indigo-700 mb-2">
                  📁 {files.length} file{files.length > 1 ? "s" : ""} selected
                </p>
                <div className="max-h-32 overflow-y-auto space-y-1">
                  {files.map((f, i) => (
                    <p key={i} className="text-xs text-gray-600 font-mono">
                      {f.webkitRelativePath || f.name}
                    </p>
                  ))}
                </div>
              </div>
            )}
            {error && <FilePreviewError />}

            {/* Options */}
            <div className="flex items-center gap-3 mb-6">
              <label className="flex items-center gap-2 text-sm text-gray-600 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableRemediation}
                  onChange={(e) => setEnableRemediation(e.target.checked)}
                  className="w-4 h-4 text-indigo-600 rounded"
                />
                Enable AI Remediation (LLM)
              </label>
            </div>

            {/* Scan Button */}
            <button
              onClick={handleScan}
              disabled={scanning || files.length === 0}
              className={`w-full font-semibold py-3 rounded-xl transition transform hover:scale-105 hover:shadow-lg ${
                scanning || files.length === 0
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-indigo-600 hover:bg-indigo-700 text-white"
              }`}
            >
              {scanning ? (
                <span className="flex items-center justify-center gap-2">
                  <svg
                    className="animate-spin h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    ></path>
                  </svg>
                  Scanning...
                </span>
              ) : (
                "Scan Configuration"
              )}
            </button>
          </div>

          {/* Results */}
          <ResultsDisplay results={results} />
        </div>
      </main>

      {/* Toast */}
      {toast === "success" && !scanning && <ToastSuccess />}
      {toast === "error" && <ToastError />}
    </div>
  );
}
