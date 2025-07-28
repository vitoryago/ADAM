import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { projectsApi } from '../lib/api';
import type { Project } from '../lib/api';
import { ProjectCard } from './ProjectCard';
import { CreateProjectModal } from './CreateProjectModal';

export function ProjectList() {
  const navigate = useNavigate();
  // State to hold our projects
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);

  // Fetch projects when component mounts
  useEffect(() => {
    console.log('ProjectList mounted, fetching projects...');
    fetchProjects();
  }, []); // Empty array means "run once on mount"

  const fetchProjects = async () => {
    try {
      console.log('Starting to fetch projects...');
      setLoading(true);
      const response = await projectsApi.list();
      console.log('Projects response:', response);
      setProjects(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to load projects');
      console.error('Error fetching projects:', err);
    } finally {
      setLoading(false);
    }
  };

  // Show loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-400">Loading projects...</div>
      </div>
    );
  }

  // Show error state
  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-400 mb-4">{error}</p>
        <button
          onClick={fetchProjects}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
        >
          Try Again
        </button>
      </div>
    );
  }

  // Show empty state
  if (projects.length === 0) {
    return (
      <>
        <div className="text-center py-12">
          <svg className="w-16 h-16 mx-auto text-gray-600 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m-9 1V7a2 2 0 012-2h6l2 2h6a2 2 0 012 2v8a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
          </svg>
          <p className="text-gray-500 mb-4">No projects yet</p>
          <button 
            onClick={() => setShowCreateModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded"
          >
            Create Your First Project
          </button>
        </div>
        <CreateProjectModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          onProjectCreated={fetchProjects}
        />
      </>
    );
  }

  // Show projects
  return (
    <>
      <div className="mb-6 flex justify-between items-center">
        <h2 className="text-xl font-semibold text-white">All Projects</h2>
        <button
          onClick={() => setShowCreateModal(true)}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span>New Project</span>
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map((project) => (
          <ProjectCard
            key={project.id}
            project={project}
            onClick={() => {
              navigate(`/project/${project.id}`);
            }}
          />
        ))}
      </div>
      <CreateProjectModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onProjectCreated={fetchProjects}
      />
    </>
  );
}