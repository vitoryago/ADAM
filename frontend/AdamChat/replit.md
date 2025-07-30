# ADAM AI Assistant - replit.md

## Overview

This is a modern project-based AI assistant application built with React, Express, and PostgreSQL, integrated with a sophisticated Python backend. ADAM (Advanced Data Analytics Model) features a hierarchical project system similar to Claude, where users can create projects with their own memory and context, then have conversations within those projects. The application uses real-time WebSocket communication, a full-stack TypeScript architecture with shared schemas, and connects to an advanced Python AI system with RAG capabilities, memory networks, and multi-model LLM integration.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: React 18 with TypeScript
- **Build Tool**: Vite for fast development and optimized builds
- **Styling**: Tailwind CSS with custom design system
- **UI Components**: Radix UI primitives with shadcn/ui components
- **State Management**: TanStack Query for server state management
- **Routing**: Wouter for lightweight client-side routing
- **Theme**: Dark/light mode support with custom theme provider

### Backend Architecture
- **Runtime**: Node.js with Express.js
- **Language**: TypeScript with ES modules
- **Database**: PostgreSQL with Drizzle ORM
- **Real-time**: WebSocket server for live chat functionality
- **Session Storage**: In-memory storage with interface for future database integration

### Database Design
- **ORM**: Drizzle with PostgreSQL dialect
- **Schema Location**: `shared/schema.ts` for type safety across frontend and backend
- **Tables**:
  - `users`: User authentication and profiles
  - `projects`: Project containers with their own memory/context
  - `conversations`: Chat conversations belonging to specific projects
  - `messages`: Individual chat messages with role-based content (user/assistant)

## Key Components

### Project Management
- **Project Dashboard**: Main interface for creating and selecting projects
- **Project Selector**: Grid-based project browser with creation dialog
- **Project Header**: Shows current project context with navigation
- **Project Memory**: Each project maintains its own context and memory

### Chat Interface
- **Chat Area**: Main conversation view with message bubbles within project context
- **Message Input**: Auto-resizing textarea with character limits and keyboard shortcuts
- **Sidebar**: Project-specific conversation history and navigation
- **Real-time Updates**: WebSocket integration for live message delivery
- **Typing Indicators**: Visual feedback during AI response generation

### UI System
- **Component Library**: Complete shadcn/ui implementation
- **Responsive Design**: Mobile-first approach with collapsible sidebar
- **Theme Support**: CSS variables for consistent theming
- **Accessibility**: ARIA labels and keyboard navigation support

### Storage Layer
- **Interface-based Design**: `IStorage` interface allows switching between memory and database storage
- **Current Implementation**: PostgreSQL database storage with Drizzle ORM
- **Database Integration**: Full PostgreSQL integration with environment variable configuration

### ADAM Python Backend Integration
- **Complete Package Structure**: Organized src/adam/ package with proper imports
- **Advanced RAG System**: Three-stage retrieval (BM25, Vector, Graph traversal)
- **Memory Networks**: Graph-based memory connections with temporal scoring
- **Multi-model LLM Support**: Intelligent routing between Grok-4, Grok-3-Mini, and O4-Mini
- **Project-Based Memory**: Isolated memory spaces per project with ChromaDB collections
- **Screen Vision Capabilities**: Optional screen capture and OCR for coworker features
- **Cost Monitoring**: Real-time usage tracking with budget limits and model optimization
- **LangGraph Integration**: State machine for conversation flow management
- **Communication Bridge**: Node.js ↔ Python via stdin/stdout JSON messaging
- **Unified LLM Client**: Supports xAI (Grok), OpenAI, and Anthropic with automatic routing

## Data Flow

1. **Project Selection**: Users select or create projects from the dashboard
2. **Project Context**: Each project maintains its own memory and conversation history
3. **User Input**: Messages entered through the chat input component within project context
4. **WebSocket Transmission**: Real-time message sending via WebSocket connection
5. **Server Processing**: Message validation, storage, and AI response simulation with project context
6. **Response Generation**: Simulated typing indicators and AI responses
7. **UI Updates**: Real-time message display and conversation state updates
8. **Persistence**: Project, conversation, and message data stored in PostgreSQL database

## External Dependencies

### Core Dependencies
- **@neondatabase/serverless**: PostgreSQL adapter for serverless environments
- **drizzle-orm**: Type-safe ORM for database operations
- **@tanstack/react-query**: Server state management and caching
- **ws**: WebSocket implementation for real-time communication

### UI Dependencies
- **@radix-ui/***: Accessible UI primitives
- **tailwindcss**: Utility-first CSS framework
- **class-variance-authority**: Type-safe component variants
- **lucide-react**: Icon library

### Development Tools
- **tsx**: TypeScript execution for development
- **esbuild**: Fast JavaScript bundler for production builds
- **drizzle-kit**: Database migration and schema management

## Deployment Strategy

### Build Process
- **Frontend**: Vite builds optimized static assets to `dist/public`
- **Backend**: esbuild bundles server code to `dist/index.js`
- **Database**: Drizzle migrations ready for PostgreSQL deployment

### Environment Configuration
- **Development**: Local development with hot reloading via Vite
- **Production**: Static file serving with Express for SPA routing
- **Database**: Configurable via `DATABASE_URL` environment variable

### Scripts
- `npm run dev`: Development server with hot reloading
- `npm run build`: Production build for both frontend and backend
- `npm run start`: Production server
- `npm run db:push`: Apply database schema changes

The application is designed to be easily deployable to platforms like Replit, Vercel, or Railway with minimal configuration changes.

## Recent Changes

### 2025-01-30 - Full ADAM RAG System Integration
- **Connected Real LLM APIs**: Integrated OpenAI GPT-4o-mini and Grok models with user's API keys
- **Implemented Cost Monitoring**: Real-time cost tracking with budget limits ($1 daily, $30 monthly)
- **Added Project-Aware Memory**: Each project gets isolated memory storage and ChromaDB collections  
- **Built Query Analysis System**: Automatic complexity detection and model routing
- **Created Unified LLM Client**: Supports OpenAI, Anthropic, and xAI (Grok) with intelligent fallbacks
- **Added Project Management**: Full project lifecycle with statistics and memory isolation