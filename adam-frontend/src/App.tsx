import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useState } from 'react';
import { ProjectList } from './components/ProjectList';
import { ConversationView } from './components/ConversationView';
import { SettingsModal } from './components/SettingsModal';

function App() {
  const [showSettings, setShowSettings] = useState(false);

  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        {/* Header */}
        <header className="bg-gray-800 border-b border-gray-700">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <a href="/" className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <span className="text-white font-bold">A</span>
                  </div>
                  <span className="text-xl font-semibold text-white">ADAM v2.0</span>
                </a>
              </div>
              
              <nav className="flex items-center space-x-4">
                <button 
                  onClick={() => setShowSettings(true)}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  Settings
                </button>
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="h-[calc(100vh-4rem)]">
          <Routes>
            <Route path="/" element={
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                <div className="mb-8">
                  <h1 className="text-3xl font-bold text-white mb-2">
                    Your Projects
                  </h1>
                  <p className="text-gray-400">
                    Select a project to start a conversation
                  </p>
                </div>
                <ProjectList />
              </div>
            } />
            <Route path="/project/:projectId" element={<ConversationView />} />
          </Routes>
        </main>
        
        <SettingsModal 
          isOpen={showSettings}
          onClose={() => setShowSettings(false)}
        />
      </div>
    </Router>
  );
}

export default App;