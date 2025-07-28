import { useState } from 'react';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [apiEndpoint, setApiEndpoint] = useState('http://localhost:8000');
  const [theme, setTheme] = useState('dark');

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-96 max-h-[80vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-semibold text-white">Settings</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
        
        <div className="space-y-4">
          {/* API Endpoint */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              API Endpoint
            </label>
            <input
              type="text"
              value={apiEndpoint}
              onChange={(e) => setApiEndpoint(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="http://localhost:8000"
            />
            <p className="text-xs text-gray-400 mt-1">
              The backend API server address
            </p>
          </div>
          
          {/* Theme */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Theme
            </label>
            <select
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="dark">Dark</option>
              <option value="light" disabled>Light (Coming Soon)</option>
            </select>
          </div>
          
          {/* About */}
          <div className="pt-4 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-2">About</h4>
            <p className="text-sm text-gray-400">
              ADAM v2.0 - Project-Based AI Assistant
            </p>
            <p className="text-xs text-gray-500 mt-1">
              Built with React, FastAPI, and ChromaDB
            </p>
          </div>
          
          {/* Keyboard Shortcuts */}
          <div className="pt-4 border-t border-gray-700">
            <h4 className="text-sm font-medium text-gray-300 mb-2">Keyboard Shortcuts</h4>
            <div className="space-y-1 text-sm text-gray-400">
              <div className="flex justify-between">
                <span>Send Message</span>
                <kbd className="bg-gray-700 px-2 py-1 rounded text-xs">Enter</kbd>
              </div>
              <div className="flex justify-between">
                <span>New Line</span>
                <kbd className="bg-gray-700 px-2 py-1 rounded text-xs">Shift + Enter</kbd>
              </div>
            </div>
          </div>
        </div>
        
        <div className="flex justify-end mt-6">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}