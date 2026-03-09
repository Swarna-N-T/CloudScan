// FooterNav.jsx
import React from "react";
import { Link } from "react-router-dom";

export default function FooterNav() {
  return (
    <footer className="bg-background-light border-t p-2">
      <nav className="flex justify-around">
        <Link to="/" className="flex flex-col items-center justify-center text-primary w-full py-1">
          <span className="material-symbols-outlined">home</span>
          <span className="text-xs font-medium">Home</span>
        </Link>

        <Link to="/upload" className="flex flex-col items-center justify-center text-gray-500 w-full py-1">
          <span className="material-symbols-outlined">add_box</span>
          <span className="text-xs font-medium">Add</span>
        </Link>

        <Link to="/settings" className="flex flex-col items-center justify-center text-gray-500 w-full py-1">
          <span className="material-symbols-outlined">settings</span>
          <span className="text-xs font-medium">Settings</span>
        </Link>
      </nav>
    </footer>
  );
}
