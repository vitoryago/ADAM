import { useContext } from "react";
import { ThemeProviderContext, type Theme, type ThemeProviderContextType } from "@/components/theme-provider";

// This hook provides a convenient way to access theme context
// It's a wrapper around the ThemeProvider context for better developer experience
export function useTheme(): ThemeProviderContextType {
  const context = useContext(ThemeProviderContext);
  
  if (!context) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  
  return context;
}

// Export the context type for TypeScript support
export type { Theme };
