// App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import UploadPage from "./pages/UploadPage";
import Settings from "./pages/Settings";
import FooterNav from "./components/FooterNav";

function App() {
  return (
    <Router>
      <div className="min-h-screen pb-20"> {/* padding bottom to not hide content under footer */}
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>

      <FooterNav />
    </Router>
  );
}

export default App;
