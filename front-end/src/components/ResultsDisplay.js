import React from "react";

/**
 * ResultsDisplay — renders the scan results returned from the CloudScan API.
 * Shows graph stats, risk summary, flagged resources, and LLM remediation.
 */

const RISK_COLORS = {
  Safe: { bg: "bg-green-100", text: "text-green-700", border: "border-green-300" },
  Low: { bg: "bg-yellow-100", text: "text-yellow-700", border: "border-yellow-300" },
  Medium: { bg: "bg-orange-100", text: "text-orange-700", border: "border-orange-300" },
  "High/Critical": { bg: "bg-red-100", text: "text-red-700", border: "border-red-300" },
};

function StatCard({ icon, label, value, color = "indigo" }) {
  return (
    <div className="bg-white rounded-xl shadow p-4 flex items-center gap-4">
      <span className={`material-symbols-outlined text-3xl text-${color}-500`}>
        {icon}
      </span>
      <div>
        <p className="text-sm text-gray-500">{label}</p>
        <p className="text-xl font-bold text-gray-800">{value}</p>
      </div>
    </div>
  );
}

function RiskBadge({ label }) {
  const colors = RISK_COLORS[label] || RISK_COLORS.Safe;
  return (
    <span
      className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${colors.bg} ${colors.text}`}
    >
      {label}
    </span>
  );
}

export default function ResultsDisplay({ results }) {
  if (!results) return null;

  const { graph_stats, risk_summary, flagged_resources, remediation, files_received } =
    results;

  return (
    <div className="mt-10 space-y-8 animate-fade-in">
      {/* ── Section Title ────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <span className="material-symbols-outlined text-indigo-500 text-3xl">
          analytics
        </span>
        <h2 className="text-2xl font-bold text-gray-800">Scan Results</h2>
      </div>

      {/* ── Graph Stats ──────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon="description" label="Files Scanned" value={files_received} />
        <StatCard icon="hub" label="Nodes" value={graph_stats?.nodes ?? 0} />
        <StatCard icon="share" label="Edges" value={graph_stats?.edges ?? 0} />
        <StatCard
          icon="category"
          label="Resource Types"
          value={graph_stats?.node_types?.length ?? 0}
        />
      </div>

      {/* ── Risk Distribution ────────────────────────────────── */}
      <div className="bg-white rounded-xl shadow p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
          <span className="material-symbols-outlined text-indigo-400">
            equalizer
          </span>
          Risk Distribution
        </h3>
        <div className="flex flex-wrap gap-4">
          {risk_summary &&
            Object.entries(risk_summary).map(([label, count]) => {
              const colors = RISK_COLORS[label] || RISK_COLORS.Safe;
              return (
                <div
                  key={label}
                  className={`flex-1 min-w-[120px] rounded-lg p-4 border ${colors.border} ${colors.bg}`}
                >
                  <p className={`text-2xl font-bold ${colors.text}`}>{count}</p>
                  <p className={`text-sm ${colors.text} opacity-80`}>{label}</p>
                </div>
              );
            })}
        </div>
      </div>

      {/* ── Flagged Resources ────────────────────────────────── */}
      {flagged_resources && flagged_resources.length > 0 ? (
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-red-400">flag</span>
            Flagged Resources ({flagged_resources.length})
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b text-gray-500">
                  <th className="py-3 px-4">#</th>
                  <th className="py-3 px-4">Resource Type</th>
                  <th className="py-3 px-4">Resource ID</th>
                  <th className="py-3 px-4">Risk Level</th>
                </tr>
              </thead>
              <tbody>
                {flagged_resources.map((res, i) => (
                  <tr
                    key={i}
                    className="border-b hover:bg-gray-50 transition-colors"
                  >
                    <td className="py-3 px-4 text-gray-500">{i + 1}</td>
                    <td className="py-3 px-4 font-mono text-sm">
                      {res.resource_type}
                    </td>
                    <td className="py-3 px-4 font-mono text-xs text-gray-600 max-w-xs truncate">
                      {res.node_id}
                    </td>
                    <td className="py-3 px-4">
                      <RiskBadge label={res.risk_label} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-green-50 border border-green-200 rounded-xl p-6 text-center">
          <span className="material-symbols-outlined text-green-500 text-4xl mb-2">
            verified
          </span>
          <p className="text-green-700 font-semibold">
            No resources were flagged — your configuration looks good!
          </p>
        </div>
      )}

      {/* ── LLM Remediation ──────────────────────────────────── */}
      {remediation && (
        <div className="bg-white rounded-xl shadow p-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined text-purple-400">
              smart_toy
            </span>
            AI Remediation Advice
          </h3>
          <div className="bg-gray-50 rounded-lg p-5 prose prose-sm max-w-none whitespace-pre-wrap text-gray-700 font-mono text-xs leading-relaxed overflow-x-auto">
            {remediation}
          </div>
        </div>
      )}
    </div>
  );
}
