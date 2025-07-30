import { useState } from "react";
import { useLocation } from "wouter";
import { useTheme } from "@/components/theme-provider";
import { ProjectSelector } from "@/components/project/project-selector";
import { Button } from "@/components/ui/button";
import { Moon, Sun } from "lucide-react";
import type { Project } from "@shared/schema";

export default function ProjectDashboard() {
  const [, setLocation] = useLocation();
  const { theme, toggleTheme } = useTheme();
  const [currentProject, setCurrentProject] = useState<Project | null>(null);

  const handleProjectSelect = (project: Project) => {
    setCurrentProject(project);
    setLocation(`/project/${project.id}`);
  };

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Top Header */}
      <div className="flex justify-end p-4 border-b border-border">
        <Button variant="ghost" size="icon" onClick={toggleTheme}>
          {theme === "light" ? (
            <Moon className="w-4 h-4" />
          ) : (
            <Sun className="w-4 h-4" />
          )}
        </Button>
      </div>

      {/* Main Content */}
      <div className="flex-1 max-w-4xl mx-auto w-full">
        <ProjectSelector
          currentProject={currentProject}
          onProjectSelect={handleProjectSelect}
        />
      </div>
    </div>
  );
}