import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import NotFound from "@/pages/not-found";
import ProjectDashboard from "./pages/project-dashboard";
import Chat from "@/pages/chat";
import TestMarkdown from "@/pages/test-markdown";
import { TestHoverPage } from "@/pages/test-hover";
import { SidebarTest } from "@/components/chat/sidebar-test";
import { TestMessagePage } from "@/pages/test-message";

function Router() {
  return (
    <Switch>
      <Route path="/" component={ProjectDashboard} />
      <Route path="/projects" component={ProjectDashboard} />
      <Route path="/test-markdown" component={TestMarkdown} />
      <Route path="/test-hover" component={TestHoverPage} />
      <Route path="/test-sidebar" component={SidebarTest} />
      <Route path="/test-message" component={TestMessagePage} />
      <Route path="/project/:projectId" component={Chat} />
      <Route path="/project/:projectId/chat/:conversationId?" component={Chat} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
