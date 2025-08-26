import type { Express } from "express";
import { createServer, type Server } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { storage } from "./storage";
import { insertProjectSchema, insertConversationSchema, insertMessageSchema } from "@shared/schema";
import { z } from "zod";
import { getADAMBridge, ADAMRequest, ADAMResponse } from "./adam-integration";

export async function registerRoutes(app: Express): Promise<Server> {
  const httpServer = createServer(app);
  
  // WebSocket server for real-time chat
  const wss = new WebSocketServer({ server: httpServer, path: '/ws' });
  
  // Store active connections
  const connections = new Map<string, WebSocket>();
  
  wss.on('connection', (ws) => {
    console.log('New WebSocket connection');
    
    ws.on('message', async (data) => {
      try {
        const message = JSON.parse(data.toString());
        
        if (message.type === 'chat') {
          // Create user message
          const userMessage = await storage.createMessage({
            conversationId: message.conversationId,
            content: message.content,
            role: 'user'
          });
          
          // Broadcast user message
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              type: 'message',
              message: userMessage
            }));
          }
          
          // Show typing indicator
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
              type: 'typing',
              isTyping: true
            }));
          }
          
          // Process through ADAM backend
          try {
            // Get conversation and project context
            const conversation = await storage.getConversation(message.conversationId);
            const project = conversation ? await storage.getProject(conversation.projectId) : null;
            
            // Get recent messages for context
            const recentMessages = await storage.getMessages(message.conversationId);
            const previousMessages = recentMessages
              .slice(-10) // Last 10 messages
              .map(msg => ({
                role: msg.role,
                content: msg.content
              }));
            
            // Prepare ADAM request
            const adamRequest: ADAMRequest = {
              query: message.content,
              conversationId: message.conversationId,
              projectId: conversation?.projectId || 'default',
              userId: conversation?.userId || 'anonymous',
              context: {
                previousMessages,
                projectMemory: project?.memory || '',
                userPreferences: {} as Record<string, any>
              }
            };
            
            // Get ADAM bridge and process query
            const adamBridge = getADAMBridge();
            const adamResponse: ADAMResponse = await adamBridge.processQuery(adamRequest);
            
            // Create AI message with ADAM response
            const aiMessage = await storage.createMessage({
              conversationId: message.conversationId,
              content: adamResponse.response,
              role: 'assistant'
            });
            
            // Stop typing indicator and send response
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'typing',
                isTyping: false
              }));
              
              ws.send(JSON.stringify({
                type: 'message',
                message: aiMessage,
                metadata: {
                  cost: adamResponse.cost,
                  modelUsed: adamResponse.modelUsed,
                  processingTime: adamResponse.processingTime,
                  memoryConfidence: adamResponse.memoryConfidence,
                  sources: adamResponse.sources,
                  conversationState: adamResponse.conversationState
                }
              }));
            }
            
            // Update project memory if this was a valuable interaction
            if (project && adamResponse.conversationState.shouldStore) {
              const updatedMemory = `${project.memory || ''}\n\nRecent interaction: ${message.content} -> ${adamResponse.response.substring(0, 200)}...`.trim();
              await storage.updateProject(project.id, {
                memory: updatedMemory,
                updatedAt: new Date()
              });
            }
            
          } catch (error) {
            console.error('ADAM processing error:', error);
            
            // Stop typing indicator
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({
                type: 'typing',
                isTyping: false
              }));
              
              // Send fallback response
              const fallbackMessage = await storage.createMessage({
                conversationId: message.conversationId,
                content: "I apologize, but I'm experiencing technical difficulties. Please try your question again.",
                role: 'assistant'
              });
              
              ws.send(JSON.stringify({
                type: 'message',
                message: fallbackMessage,
                metadata: {
                  error: true,
                  cost: 0,
                  modelUsed: 'fallback'
                }
              }));
            }
          }
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: 'error',
            error: 'Invalid message format'
          }));
        }
      }
    });
    
    ws.on('close', () => {
      console.log('WebSocket connection closed');
    });
    
    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  });
  
  // API Routes
  
  // Project routes
  app.get("/api/projects", async (_req, res) => {
    try {
      const projects = await storage.getProjects();
      res.json(projects);
    } catch (error) {
      console.error("Failed to get projects:", error);
      res.status(500).json({ message: "Failed to get projects" });
    }
  });
  
  app.post("/api/projects", async (req, res) => {
    try {
      const validatedData = insertProjectSchema.parse(req.body);
      const project = await storage.createProject(validatedData);
      res.status(201).json(project);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid project data", errors: error.errors });
      } else {
        console.error("Failed to create project:", error);
        res.status(500).json({ message: "Failed to create project" });
      }
    }
  });
  
  app.get("/api/projects/:id", async (req, res) => {
    try {
      const project = await storage.getProject(req.params.id);
      if (!project) {
        return res.status(404).json({ message: "Project not found" });
      }
      res.json(project);
    } catch (error) {
      console.error("Failed to get project:", error);
      res.status(500).json({ message: "Failed to get project" });
    }
  });
  
  app.delete("/api/projects/:id", async (req, res) => {
    try {
      const deleted = await storage.deleteProject(req.params.id);
      if (!deleted) {
        return res.status(404).json({ message: "Project not found" });
      }
      res.status(204).send();
    } catch (error) {
      console.error("Failed to delete project:", error);
      res.status(500).json({ message: "Failed to delete project" });
    }
  });
  
  // Get conversations for a project
  app.get("/api/projects/:id/conversations", async (req, res) => {
    try {
      const conversations = await storage.getConversations(req.params.id);
      res.json(conversations);
    } catch (error) {
      console.error("Failed to get conversations:", error);
      res.status(500).json({ message: "Failed to get conversations" });
    }
  });

  // Create new conversation in a project
  app.post("/api/projects/:id/conversations", async (req, res) => {
    try {
      const validatedData = insertConversationSchema.parse({
        ...req.body,
        projectId: req.params.id
      });
      const conversation = await storage.createConversation(validatedData);
      res.status(201).json(conversation);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid conversation data", errors: error.errors });
      } else {
        console.error("Failed to create conversation:", error);
        res.status(500).json({ message: "Failed to create conversation" });
      }
    }
  });
  
  // Get all conversations (deprecated, but keeping for backward compatibility)
  app.get("/api/conversations", async (_req, res) => {
    try {
      const conversations = await storage.getConversations();
      res.json(conversations);
    } catch (error) {
      console.error("Failed to get conversations:", error);
      res.status(500).json({ message: "Failed to get conversations" });
    }
  });
  
  // Create new conversation
  app.post("/api/conversations", async (req, res) => {
    try {
      const validatedData = insertConversationSchema.parse(req.body);
      const conversation = await storage.createConversation(validatedData);
      res.status(201).json(conversation);
    } catch (error) {
      if (error instanceof z.ZodError) {
        res.status(400).json({ message: "Invalid conversation data", errors: error.errors });
      } else {
        console.error("Failed to create conversation:", error);
        res.status(500).json({ message: "Failed to create conversation" });
      }
    }
  });
  
  // Get conversation by ID
  app.get("/api/conversations/:id", async (req, res) => {
    try {
      const conversation = await storage.getConversation(req.params.id);
      if (!conversation) {
        return res.status(404).json({ message: "Conversation not found" });
      }
      res.json(conversation);
    } catch (error) {
      console.error("Failed to get conversation:", error);
      res.status(500).json({ message: "Failed to get conversation" });
    }
  });
  
  // Update conversation title
  app.patch("/api/conversations/:id", async (req, res) => {
    try {
      const { title } = req.body;
      if (!title || typeof title !== 'string') {
        return res.status(400).json({ message: "Title is required" });
      }
      
      const conversation = await storage.getConversation(req.params.id);
      if (!conversation) {
        return res.status(404).json({ message: "Conversation not found" });
      }
      
      const updated = await storage.updateConversation(req.params.id, { 
        title: title.trim(),
        updatedAt: new Date()
      });
      
      res.json(updated);
    } catch (error) {
      console.error("Failed to update conversation:", error);
      res.status(500).json({ message: "Failed to update conversation" });
    }
  });

  // Delete conversation
  app.delete("/api/conversations/:id", async (req, res) => {
    try {
      const deleted = await storage.deleteConversation(req.params.id);
      if (!deleted) {
        return res.status(404).json({ message: "Conversation not found" });
      }
      res.status(204).send();
    } catch (error) {
      console.error("Failed to delete conversation:", error);
      res.status(500).json({ message: "Failed to delete conversation" });
    }
  });
  
  // Get messages for a conversation
  app.get("/api/conversations/:id/messages", async (req, res) => {
    try {
      const messages = await storage.getMessages(req.params.id);
      res.json(messages);
    } catch (error) {
      console.error("Failed to get messages:", error);
      res.status(500).json({ message: "Failed to get messages" });
    }
  });

  // Create message with streaming response
  app.post("/api/conversations/:id/messages/stream", async (req, res) => {
    try {
      const { content, use_memory, model, use_search, search_mode, has_image, image_data } = req.body;
      
      // Create user message
      const userMessage = await storage.createMessage({
        conversationId: req.params.id,
        content,
        role: 'user'
      });
      
      // Setup SSE response
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });
      
      // Get conversation and project context
      const conversation = await storage.getConversation(req.params.id);
      const project = conversation ? await storage.getProject(conversation.projectId) : null;
      
      // Get recent messages for context
      const recentMessages = await storage.getMessages(req.params.id);
      const previousMessages = recentMessages
        .slice(-10)
        .map(msg => ({
          role: msg.role,
          content: msg.content
        }));
      
      try {
        // Prepare ADAM request
        const adamRequest: ADAMRequest = {
          query: content,
          conversationId: req.params.id,
          projectId: conversation?.projectId || 'default',
          userId: conversation?.userId || 'anonymous',
          context: {
            previousMessages,
            projectMemory: project?.memory || '',
            userPreferences: { 
              model,
              use_search,
              search_mode,
              has_image,
              image_data 
            }
          }
        };
        
        // Get ADAM bridge and process query
        const adamBridge = getADAMBridge();
        const adamResponse: ADAMResponse = await adamBridge.processQuery(adamRequest);
        
        // Stream the response in chunks
        const chunks = adamResponse.response.match(/.{1,50}/g) || [];
        for (const chunk of chunks) {
          res.write(`data: ${JSON.stringify({ type: 'assistant_chunk', content: chunk })}\n\n`);
          await new Promise(resolve => setTimeout(resolve, 50)); // Simulate streaming
        }
        
        // Send completion event
        res.write(`data: ${JSON.stringify({ 
          type: 'complete',
          model: adamResponse.modelUsed,
          tokens: adamResponse.processingTime,
          cost: adamResponse.cost
        })}\n\n`);
        
        // Save assistant message
        await storage.createMessage({
          conversationId: req.params.id,
          content: adamResponse.response,
          role: 'assistant'
        });
        
        // Update project memory if valuable
        if (project && adamResponse.conversationState.shouldStore) {
          const updatedMemory = `${project.memory || ''}\n\nRecent: ${content.substring(0, 100)}... -> ${adamResponse.response.substring(0, 100)}...`.trim();
          await storage.updateProject(project.id, { memory: updatedMemory });
        }
        
      } catch (error) {
        console.error('ADAM processing error:', error);
        res.write(`data: ${JSON.stringify({ 
          type: 'error', 
          message: 'Failed to process message' 
        })}\n\n`);
      }
      
      res.end();
    } catch (error) {
      console.error("Failed to create streaming message:", error);
      res.status(500).json({ message: "Failed to create message" });
    }
  });
  
  // Clear all conversations in a project
  app.delete("/api/projects/:id/conversations", async (req, res) => {
    try {
      const conversations = await storage.getConversations(req.params.id);
      await Promise.all(conversations.map(conv => storage.deleteConversation(conv.id)));
      res.status(204).send();
    } catch (error) {
      console.error("Failed to clear conversations:", error);
      res.status(500).json({ message: "Failed to clear conversations" });
    }
  });

  // Clear all conversations (deprecated)
  app.delete("/api/conversations", async (_req, res) => {
    try {
      const conversations = await storage.getConversations();
      await Promise.all(conversations.map(conv => storage.deleteConversation(conv.id)));
      res.status(204).send();
    } catch (error) {
      console.error("Failed to clear conversations:", error);
      res.status(500).json({ message: "Failed to clear conversations" });
    }
  });

  // ADAM cost monitoring endpoint
  app.get("/api/adam/cost-summary", async (_req, res) => {
    try {
      const adamBridge = getADAMBridge();
      const costSummary = await adamBridge.getCostSummary();
      res.json(costSummary);
    } catch (error) {
      console.error("Failed to get ADAM cost summary:", error);
      res.status(500).json({ 
        message: "Failed to get cost summary",
        dailyCost: 0.0,
        monthlyCost: 0.0,
        queryCount: 0,
        modelUsage: {}
      });
    }
  });

  return httpServer;
}


