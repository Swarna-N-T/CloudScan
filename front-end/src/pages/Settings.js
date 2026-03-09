// Settings.jsx
import React, { useEffect, useState } from "react";
import { ToastSuccess } from "../components/Toast";

export default function Settings() {
  const [toast, setToast] = useState(false);

  const [settings, setSettings] = useState({
    llmKey: "",
    neo4jUri: "",
    neo4jUser: "",
    neo4jPassword: "",
    scanType: "basic",
    enableML: true,
    enableGraph: true,
    autoDelete: false,
    fontSize: "medium",
  });

  /* ================= APPLY FONT SIZE GLOBALLY ================= */
  useEffect(() => {
    document.documentElement.classList.remove(
      "text-sm",
      "text-base",
      "text-lg"
    );

    if (settings.fontSize === "small") {
      document.documentElement.classList.add("text-sm");
    } else if (settings.fontSize === "large") {
      document.documentElement.classList.add("text-lg");
    } else {
      document.documentElement.classList.add("text-base");
    }

    localStorage.setItem("fontSize", settings.fontSize);
  }, [settings.fontSize]);

  /* ================= LOAD SAVED FONT SIZE ================= */
  useEffect(() => {
    const savedFontSize = localStorage.getItem("fontSize");
    if (savedFontSize) {
      setSettings((prev) => ({
        ...prev,
        fontSize: savedFontSize,
      }));
    }
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings({
      ...settings,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const saveSettings = () => {
    setToast(true);
    setTimeout(() => setToast(false), 3000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 to-blue-100 p-8">

      {/* PAGE TITLE */}
      <div className="max-w-4xl mx-auto mb-10">
        <h1 className="text-3xl font-bold text-indigo-700 mb-2">
          Settings
        </h1>
        <p className="text-gray-600">
          Configure integrations, scanning behavior, privacy, and appearance options.
        </p>
      </div>

      <div className="max-w-4xl mx-auto space-y-8">

        {/* API & INTEGRATIONS */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">API & Integrations</h2>

          <input
            type="password"
            name="llmKey"
            placeholder="LLM API Key"
            className="w-full p-3 border rounded-lg mb-3"
            onChange={handleChange}
          />

          <input
            type="text"
            name="neo4jUri"
            placeholder="Neo4j URI"
            className="w-full p-3 border rounded-lg mb-3"
            onChange={handleChange}
          />

          <input
            type="text"
            name="neo4jUser"
            placeholder="Neo4j Username"
            className="w-full p-3 border rounded-lg mb-3"
            onChange={handleChange}
          />

          <input
            type="password"
            name="neo4jPassword"
            placeholder="Neo4j Password"
            className="w-full p-3 border rounded-lg"
            onChange={handleChange}
          />
        </div>

        {/* SCAN PREFERENCES */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Scan Preferences</h2>

          <select
            name="scanType"
            value={settings.scanType}
            onChange={handleChange}
            className="w-full p-3 border rounded-lg mb-4"
          >
            <option value="basic">Basic Scan</option>
            <option value="deep">Deep Scan</option>
          </select>

          <label className="flex items-center gap-3 mb-2">
            <input
              type="checkbox"
              name="enableML"
              checked={settings.enableML}
              onChange={handleChange}
            />
            Enable ML-based Risk Prioritization
          </label>

          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              name="enableGraph"
              checked={settings.enableGraph}
              onChange={handleChange}
            />
            Enable Graph-based Analysis (Neo4j)
          </label>
        </div>

        {/* APPEARANCE */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Appearance</h2>

          <label className="block font-medium mb-2">
            Font Size
          </label>

          <select
            name="fontSize"
            value={settings.fontSize}
            onChange={handleChange}
            className="w-full p-3 border rounded-lg"
          >
            <option value="small">Small</option>
            <option value="medium">Medium (Default)</option>
            <option value="large">Large</option>
          </select>
        </div>

        {/* SECURITY & PRIVACY */}
        <div className="bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-semibold mb-4">Security & Privacy</h2>

          <label className="flex items-center gap-3">
            <input
              type="checkbox"
              name="autoDelete"
              checked={settings.autoDelete}
              onChange={handleChange}
            />
            Automatically delete uploaded files after scan
          </label>
        </div>

        {/* SAVE BUTTON */}
        <div className="flex justify-end">
          <button
            onClick={saveSettings}
            className="px-6 py-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition"
          >
            Save Settings
          </button>
        </div>
      </div>

      {toast && <ToastSuccess />}
    </div>
  );
}
