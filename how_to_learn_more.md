# React Learning Journey for ADAM v2

## Overview
This document tracks your progress learning React while building ADAM v2's frontend. Each concept is introduced as we need it, making learning practical and immediately applicable.

## Learning Path

### ✅ Phase 1: React Basics (Current)
**What you're learning:**
- Components and JSX
- Props and State
- Event handling
- Conditional rendering

**ADAM Features:**
- Project list display
- Basic navigation
- Simple forms

**Key Concepts:**
```tsx
// Component: Reusable UI piece
function ProjectCard({ project }) {
  return (
    <div className="p-4 border rounded">
      <h3>{project.name}</h3>
      <p>{project.description}</p>
    </div>
  );
}

// State: Component's memory
const [projects, setProjects] = useState([]);

// Props: Pass data to components
<ProjectCard project={myProject} />
```

### 📝 Phase 2: Data Fetching
**What you'll learn:**
- useEffect hook
- Async operations
- Loading states
- Error handling

**ADAM Features:**
- Fetch projects from API
- Show loading spinners
- Handle API errors

**Key Concepts:**
```tsx
useEffect(() => {
  fetchProjects().then(setProjects);
}, []); // Empty array = run once on mount
```

### 📝 Phase 3: Routing
**What you'll learn:**
- React Router
- Navigation
- URL parameters
- Protected routes

**ADAM Features:**
- Project pages
- Conversation views
- Navigation between screens

### 📝 Phase 4: State Management
**What you'll learn:**
- Zustand for global state
- Context API
- State patterns

**ADAM Features:**
- User preferences
- Current project/conversation
- Global settings

### 📝 Phase 5: Real-time Features
**What you'll learn:**
- WebSockets/SSE
- Streaming responses
- Real-time updates

**ADAM Features:**
- Streaming AI responses
- Live message updates
- Real-time memory stats

### 📝 Phase 6: Advanced UI
**What you'll learn:**
- Custom hooks
- Performance optimization
- Animations
- Accessibility

**ADAM Features:**
- Smooth transitions
- Keyboard shortcuts
- Screen reader support

## React Patterns You'll Master

### 1. Component Composition
```tsx
// Instead of inheritance, use composition
<Layout>
  <Header />
  <MainContent>
    <ProjectList />
  </MainContent>
  <Footer />
</Layout>
```

### 2. Custom Hooks
```tsx
// Reusable logic
function useProjects() {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    // Fetch logic here
  }, []);
  
  return { projects, loading };
}
```

### 3. Conditional Rendering
```tsx
{loading ? (
  <Spinner />
) : error ? (
  <ErrorMessage error={error} />
) : (
  <ProjectList projects={projects} />
)}
```

## TypeScript Benefits

You'll learn TypeScript alongside React:
- Type safety prevents bugs
- Better IDE support
- Self-documenting code
- Easier refactoring

Example:
```tsx
interface Project {
  id: string;
  name: string;
  description?: string; // Optional
  createdAt: Date;
}

// TypeScript catches errors at compile time
function ProjectCard({ project }: { project: Project }) {
  return <div>{project.name}</div>;
}
```

## Resources

### Documentation
- [React Docs (New)](https://react.dev) - Start here!
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)

### Tools We're Using
- **Vite**: Fast build tool
- **React Router**: Navigation
- **Tanstack Query**: Data fetching
- **Zustand**: State management
- **Axios**: HTTP client
- **React Markdown**: Render markdown

### Learning Tips
1. **Don't memorize** - Understand concepts
2. **Build, don't just read** - Practice is key
3. **Break complex problems** into small pieces
4. **Use TypeScript** - It helps you learn
5. **Ask questions** - I'm here to explain

## Progress Tracker

- [ ] Created first component
- [ ] Used props successfully
- [ ] Managed state with useState
- [ ] Fetched data from API
- [ ] Added routing
- [ ] Styled with Tailwind
- [ ] Handled user input
- [ ] Displayed markdown content
- [ ] Implemented streaming responses
- [ ] Added error boundaries
- [ ] Optimized performance
- [ ] Deployed to production

## Common Patterns in ADAM

### API Integration
```tsx
const api = axios.create({
  baseURL: 'http://localhost:8000/api',
  headers: {
    'Content-Type': 'application/json',
  },
});
```

### Streaming Responses
```tsx
const response = await fetch('/api/messages/stream');
const reader = response.body.getReader();
// Process chunks as they arrive
```

### Markdown Rendering
```tsx
<ReactMarkdown 
  remarkPlugins={[remarkGfm]}
  components={{
    code: ({ node, ...props }) => (
      <CodeBlock {...props} />
    ),
  }}
>
  {content}
</ReactMarkdown>
```

Remember: We're building this together. Each feature teaches new concepts. You're not just learning React - you're building a real, complex application!