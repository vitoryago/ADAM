import { type User, type InsertUser, type Project, type InsertProject, type Conversation, type InsertConversation, type Message, type InsertMessage, users, projects, conversations, messages } from "@shared/schema";
import { randomUUID } from "crypto";
import { db } from "./db";
import { eq } from "drizzle-orm";

export interface IStorage {
  getUser(id: string): Promise<User | undefined>;
  getUserByUsername(username: string): Promise<User | undefined>;
  createUser(user: InsertUser): Promise<User>;
  
  getProjects(userId?: string): Promise<Project[]>;
  getProject(id: string): Promise<Project | undefined>;
  createProject(project: InsertProject): Promise<Project>;
  deleteProject(id: string): Promise<boolean>;
  updateProject(id: string, updates: Partial<Project>): Promise<Project | undefined>;
  
  getConversations(projectId?: string): Promise<Conversation[]>;
  getConversation(id: string): Promise<Conversation | undefined>;
  createConversation(conversation: InsertConversation): Promise<Conversation>;
  deleteConversation(id: string): Promise<boolean>;
  updateConversation(id: string, updates: Partial<Conversation>): Promise<Conversation | undefined>;
  
  getMessages(conversationId: string): Promise<Message[]>;
  createMessage(message: InsertMessage): Promise<Message>;
  deleteMessages(conversationId: string): Promise<boolean>;
}

export class MemStorage implements IStorage {
  private users: Map<string, User>;
  private projects: Map<string, Project>;
  private conversations: Map<string, Conversation>;
  private messages: Map<string, Message>;

  constructor() {
    this.users = new Map();
    this.projects = new Map();
    this.conversations = new Map();
    this.messages = new Map();
  }

  async getUser(id: string): Promise<User | undefined> {
    return this.users.get(id);
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    return Array.from(this.users.values()).find(
      (user) => user.username === username,
    );
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    this.users.set(id, user);
    return user;
  }

  async getProjects(userId?: string): Promise<Project[]> {
    const projects = Array.from(this.projects.values());
    if (userId) {
      return projects.filter(project => project.userId === userId);
    }
    return projects.sort((a, b) => 
      new Date(b.updatedAt || b.createdAt || 0).getTime() - 
      new Date(a.updatedAt || a.createdAt || 0).getTime()
    );
  }

  async getProject(id: string): Promise<Project | undefined> {
    return this.projects.get(id);
  }

  async createProject(insertProject: InsertProject): Promise<Project> {
    const id = randomUUID();
    const now = new Date();
    const project: Project = {
      ...insertProject,
      id,
      description: insertProject.description || null,
      memory: insertProject.memory || null,
      userId: insertProject.userId || null,
      createdAt: now,
      updatedAt: now,
    };
    this.projects.set(id, project);
    return project;
  }

  async deleteProject(id: string): Promise<boolean> {
    const deleted = this.projects.delete(id);
    if (deleted) {
      // Delete all conversations in this project
      const conversations = await this.getConversations(id);
      await Promise.all(conversations.map(conv => this.deleteConversation(conv.id)));
    }
    return deleted;
  }

  async updateProject(id: string, updates: Partial<Project>): Promise<Project | undefined> {
    const project = this.projects.get(id);
    if (!project) return undefined;
    
    const updatedProject = {
      ...project,
      ...updates,
      updatedAt: new Date(),
    };
    this.projects.set(id, updatedProject);
    return updatedProject;
  }

  async getConversations(projectId?: string): Promise<Conversation[]> {
    const conversations = Array.from(this.conversations.values());
    if (projectId) {
      return conversations.filter(conv => conv.projectId === projectId);
    }
    return conversations.sort((a, b) => 
      new Date(b.updatedAt || b.createdAt || 0).getTime() - 
      new Date(a.updatedAt || a.createdAt || 0).getTime()
    );
  }

  async getConversation(id: string): Promise<Conversation | undefined> {
    return this.conversations.get(id);
  }

  async createConversation(insertConversation: InsertConversation): Promise<Conversation> {
    const id = randomUUID();
    const now = new Date();
    const conversation: Conversation = {
      ...insertConversation,
      id,
      userId: insertConversation.userId || null,
      createdAt: now,
      updatedAt: now,
    };
    this.conversations.set(id, conversation);
    return conversation;
  }

  async deleteConversation(id: string): Promise<boolean> {
    const deleted = this.conversations.delete(id);
    if (deleted) {
      // Also delete all messages in this conversation
      await this.deleteMessages(id);
    }
    return deleted;
  }

  async updateConversation(id: string, updates: Partial<Conversation>): Promise<Conversation | undefined> {
    const conversation = this.conversations.get(id);
    if (!conversation) return undefined;
    
    const updatedConversation = {
      ...conversation,
      ...updates,
      updatedAt: new Date(),
    };
    this.conversations.set(id, updatedConversation);
    return updatedConversation;
  }

  async getMessages(conversationId: string): Promise<Message[]> {
    return Array.from(this.messages.values())
      .filter(message => message.conversationId === conversationId)
      .sort((a, b) => 
        new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime()
      );
  }

  async createMessage(insertMessage: InsertMessage): Promise<Message> {
    const id = randomUUID();
    const message: Message = {
      ...insertMessage,
      id,
      timestamp: new Date(),
      metadata: null,
    };
    this.messages.set(id, message);
    
    // Update conversation timestamp
    await this.updateConversation(insertMessage.conversationId, {});
    
    return message;
  }

  async deleteMessages(conversationId: string): Promise<boolean> {
    const messagesToDelete = Array.from(this.messages.entries())
      .filter(([_, message]) => message.conversationId === conversationId);
    
    messagesToDelete.forEach(([id]) => {
      this.messages.delete(id);
    });
    
    return messagesToDelete.length > 0;
  }
}

export class DatabaseStorage implements IStorage {
  async getUser(id: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.id, id));
    return user || undefined;
  }

  async getUserByUsername(username: string): Promise<User | undefined> {
    const [user] = await db.select().from(users).where(eq(users.username, username));
    return user || undefined;
  }

  async createUser(insertUser: InsertUser): Promise<User> {
    const id = randomUUID();
    const user: User = { ...insertUser, id };
    const [createdUser] = await db
      .insert(users)
      .values(user)
      .returning();
    return createdUser;
  }

  async getProjects(userId?: string): Promise<Project[]> {
    const projectList = userId 
      ? await db.select().from(projects).where(eq(projects.userId, userId))
      : await db.select().from(projects);
    return projectList.sort((a, b) => 
      new Date(b.updatedAt || b.createdAt || 0).getTime() - 
      new Date(a.updatedAt || a.createdAt || 0).getTime()
    );
  }

  async getProject(id: string): Promise<Project | undefined> {
    const [project] = await db.select().from(projects).where(eq(projects.id, id));
    return project || undefined;
  }

  async createProject(insertProject: InsertProject): Promise<Project> {
    const id = randomUUID();
    const now = new Date();
    const project: Project = {
      ...insertProject,
      id,
      description: insertProject.description || null,
      memory: insertProject.memory || null,
      userId: insertProject.userId || null,
      createdAt: now,
      updatedAt: now,
    };
    const [createdProject] = await db
      .insert(projects)
      .values(project)
      .returning();
    return createdProject;
  }

  async deleteProject(id: string): Promise<boolean> {
    // Delete all conversations in this project first
    const projectConversations = await this.getConversations(id);
    await Promise.all(projectConversations.map(conv => this.deleteConversation(conv.id)));
    
    const result = await db.delete(projects).where(eq(projects.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  async updateProject(id: string, updates: Partial<Project>): Promise<Project | undefined> {
    const [project] = await db
      .update(projects)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(projects.id, id))
      .returning();
    return project || undefined;
  }

  async getConversations(projectId?: string): Promise<Conversation[]> {
    const conversationList = projectId
      ? await db.select().from(conversations).where(eq(conversations.projectId, projectId))
      : await db.select().from(conversations);
    return conversationList.sort((a, b) => 
      new Date(b.updatedAt || b.createdAt || 0).getTime() - 
      new Date(a.updatedAt || a.createdAt || 0).getTime()
    );
  }

  async getConversation(id: string): Promise<Conversation | undefined> {
    const [conversation] = await db.select().from(conversations).where(eq(conversations.id, id));
    return conversation || undefined;
  }

  async createConversation(insertConversation: InsertConversation): Promise<Conversation> {
    const id = randomUUID();
    const now = new Date();
    const conversation: Conversation = {
      ...insertConversation,
      id,
      userId: insertConversation.userId || null,
      createdAt: now,
      updatedAt: now,
    };
    const [createdConversation] = await db
      .insert(conversations)
      .values(conversation)
      .returning();
    return createdConversation;
  }

  async deleteConversation(id: string): Promise<boolean> {
    // Delete all messages in this conversation first
    await this.deleteMessages(id);
    
    const result = await db.delete(conversations).where(eq(conversations.id, id));
    return (result.rowCount ?? 0) > 0;
  }

  async updateConversation(id: string, updates: Partial<Conversation>): Promise<Conversation | undefined> {
    const [conversation] = await db
      .update(conversations)
      .set({ ...updates, updatedAt: new Date() })
      .where(eq(conversations.id, id))
      .returning();
    return conversation || undefined;
  }

  async getMessages(conversationId: string): Promise<Message[]> {
    const messageList = await db.select().from(messages).where(eq(messages.conversationId, conversationId));
    return messageList.sort((a, b) => 
      new Date(a.timestamp || 0).getTime() - new Date(b.timestamp || 0).getTime()
    );
  }

  async createMessage(insertMessage: InsertMessage): Promise<Message> {
    const id = randomUUID();
    const message: Message = {
      ...insertMessage,
      id,
      timestamp: new Date(),
      metadata: null,
    };
    const [createdMessage] = await db
      .insert(messages)
      .values(message)
      .returning();
    
    // Update conversation timestamp
    await this.updateConversation(insertMessage.conversationId, {});
    
    return createdMessage;
  }

  async deleteMessages(conversationId: string): Promise<boolean> {
    const result = await db.delete(messages).where(eq(messages.conversationId, conversationId));
    return (result.rowCount ?? 0) > 0;
  }
}

export const storage = new DatabaseStorage();
