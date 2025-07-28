import type { Project } from '../lib/api';

interface ProjectCardProps {
  project: Project;
  onClick: () => void;
}

export function ProjectCard({ project, onClick }: ProjectCardProps) {
  return (
    <div
      onClick={onClick}
      className="bg-gray-800 rounded-lg p-6 hover:bg-gray-700 transition-colors cursor-pointer"
    >
      <h3 className="text-xl font-semibold text-white mb-2">
        {project.name}
      </h3>
      
      {project.description && (
        <p className="text-gray-400 text-sm mb-4">
          {project.description}
        </p>
      )}
      
      <div className="flex items-center justify-between text-xs text-gray-500">
        <span>
          Created {new Date(project.created_at).toLocaleDateString()}
        </span>
        
        {project.is_archived && (
          <span className="bg-gray-700 px-2 py-1 rounded">
            Archived
          </span>
        )}
      </div>
    </div>
  );
}