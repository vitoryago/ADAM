import { Switch, Route } from "wouter";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/theme-provider";
import NotFound from "@/pages/not-found";
import ProjectDashboard from "./pages/project-dashboard";
import Chat from "@/pages/chat";
import DeepDiscussion from "@/pages/deep-discussion";

function Router() {
  return (
    <Switch>
      <Route path="/" component={ProjectDashboard} />
      <Route path="/projects" component={ProjectDashboard} />
      <Route path="/project/:projectId" component={Chat} />
      <Route path="/project/:projectId/chat/:conversationId?" component={Chat} />
      <Route path="/project/:projectId/deep-discussion" component={DeepDiscussion} />
      <Route path="/project/:projectId/deep-discussion/:sessionId?" component={DeepDiscussion} />
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
