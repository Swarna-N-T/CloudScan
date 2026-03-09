// Home.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-8">

      {/* HERO SECTION */}
      <div className="max-w-5xl mx-auto text-center mb-14">
        <h1 className="text-4xl font-bold text-indigo-700 mb-4">
          CloudScan
        </h1>
        <p className="text-lg text-gray-600">
          AI-Powered Cloud Misconfiguration Analyzer for Secure Infrastructure
        </p>

        <button
          onClick={() => navigate("/upload")}
          className="mt-6 px-8 py-3 bg-indigo-600 text-white font-semibold rounded-xl hover:bg-indigo-700 transition transform hover:scale-105"
        >
          Start Scan
        </button>
      </div>

      {/* WORKFLOW SECTION */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-6 mb-16">

        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <span className="material-symbols-outlined text-indigo-500 text-4xl mb-3">
            upload_file
          </span>
          <h3 className="font-semibold text-lg mb-2">Upload IaC</h3>
          <p className="text-sm text-gray-500">
            Upload Terraform or JSON configuration files.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <span className="material-symbols-outlined text-indigo-500 text-4xl mb-3">
            bug_report
          </span>
          <h3 className="font-semibold text-lg mb-2">Scan Resources</h3>
          <p className="text-sm text-gray-500">
            Detect misconfigurations in storage, IAM, and networking.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <span className="material-symbols-outlined text-indigo-500 text-4xl mb-3">
            psychology
          </span>
          <h3 className="font-semibold text-lg mb-2">AI Risk Analysis</h3>
          <p className="text-sm text-gray-500">
            ML prioritizes risks based on severity and context.
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-md p-6 text-center">
          <span className="material-symbols-outlined text-indigo-500 text-4xl mb-3">
            security
          </span>
          <h3 className="font-semibold text-lg mb-2">Fix Suggestions</h3>
          <p className="text-sm text-gray-500">
            Get least-privilege and security recommendations.
          </p>
        </div>
      </div>

      {/* DETECTION COVERAGE */}
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8 text-center">
        <h2 className="text-2xl font-semibold text-gray-800 mb-4">
          What CloudScan Detects
        </h2>

        <div className="flex flex-wrap justify-center gap-4 mt-4">
          <span className="px-4 py-2 bg-red-100 text-red-700 rounded-full text-sm">
            Public Storage
          </span>
          <span className="px-4 py-2 bg-yellow-100 text-yellow-700 rounded-full text-sm">
            Open Firewalls
          </span>
          <span className="px-4 py-2 bg-blue-100 text-blue-700 rounded-full text-sm">
            IAM Misuse
          </span>
          <span className="px-4 py-2 bg-purple-100 text-purple-700 rounded-full text-sm">
            Missing Encryption
          </span>
          <span className="px-4 py-2 bg-green-100 text-green-700 rounded-full text-sm">
            Over-Permissive Access
          </span>
        </div>
      </div>
    </div>
  );
}
