import { useState } from 'react';
import { projectsApi } from '../lib/api';

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
  onProjectCreated: () => void;
}

export function CreateProjectModal({ isOpen, onClose, onProjectCreated }: CreateProjectModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Project name is required');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      await projectsApi.create({
        name: name.trim(),
        description: description.trim() || undefined,
      });
      
      // Reset form
      setName('');
      setDescription('');
      
      // Notify parent
      onProjectCreated();
      onClose();
    } catch (err) {
      setError('Failed to create project');
      console.error('Error creating project:', err);
    } finally {
      setIsCreating(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-96">
        <h3 className="text-xl font-semibold text-white mb-4">Create New Project</h3>
        
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Project Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="My Awesome Project"
              autoFocus
            />
          </div>
          
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Description (Optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="A brief description of your project..."
              rows={3}
            />
          </div>
          
          {error && (
            <div className="mb-4 text-red-400 text-sm">
              {error}
            </div>
          )}
          
          <div className="flex justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
              disabled={isCreating}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors disabled:bg-gray-600"
              disabled={isCreating}
            >
              {isCreating ? 'Creating...' : 'Create Project'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}