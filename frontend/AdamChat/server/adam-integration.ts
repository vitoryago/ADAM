import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';
import fs from 'fs/promises';

/**
 * Integration layer between the Node.js web app and Python ADAM backend
 * 
 * This service manages communication with the Python ADAM system that includes:
 * - Advanced RAG with BM25, vector search, and graph traversal
 * - Memory networks with conversation awareness
 * - Multi-model LLM integration (Grok-3, O3, Claude Opus 4)
 * - Real-time cost monitoring
 */

export interface ADAMRequest {
  query: string;
  conversationId: string;
  projectId: string;
  userId: string;
  context?: {
    previousMessages?: Array<{role: string, content: string}>;
    projectMemory?: string;
    userPreferences?: Record<string, any>;
  };
}

export interface ADAMResponse {
  response: string;
  memoryId?: string;
  cost: number;
  modelUsed: string;
  processingTime: number;
  memoryConfidence?: number;
  sources?: Array<{
    id: string;
    content: string;
    similarity: number;
    method: 'bm25' | 'vector' | 'graph';
  }>;
  conversationState: {
    complexity: 'simple' | 'moderate' | 'complex';
    memoryFound: boolean;
    shouldStore: boolean;
  };
}

export interface ADAMError {
  error: string;
  code: string;
  retryable: boolean;
}

class ADAMPythonBridge extends EventEmitter {
  private pythonProcess: ChildProcess | null = null;
  private isInitialized = false;
  private requestQueue: Map<string, {
    resolve: (value: ADAMResponse) => void;
    reject: (error: ADAMError) => void;
    timeout: NodeJS.Timeout;
  }> = new Map();

  constructor(
    private pythonScriptPath: string = './python_backend/main.py',
    private pythonEnv: string = 'python3'
  ) {
    super();
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Check if Python backend files exist
      await this.validatePythonBackend();
      
      // Start Python process
      await this.startPythonProcess();
      
      this.isInitialized = true;
      console.log('ADAM Python backend initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ADAM Python backend:', error);
      throw error;
    }
  }

  private async validatePythonBackend(): Promise<void> {
    const requiredFiles = [
      'src/adam/advanced_rag.py',
      'src/adam/memory_network.py',
      'src/adam/conversation_aware_memory.py',
      'src/adam/langgraph_conversation.py',
      'src/adam/integrated_conversation_system.py',
      'src/adam/llm/client.py',
      'src/adam/project_manager.py'
    ];

    for (const file of requiredFiles) {
      const filePath = path.join(path.dirname(this.pythonScriptPath), file);
      try {
        await fs.access(filePath);
      } catch {
        throw new Error(`Required Python file not found: ${file}`);
      }
    }
  }

  private async startPythonProcess(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.pythonProcess = spawn(this.pythonEnv, [this.pythonScriptPath], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: {
          ...process.env,
          PYTHONPATH: path.dirname(this.pythonScriptPath),
          ADAM_MODE: 'service'
        }
      });

      let initBuffer = '';

      this.pythonProcess.stdout?.on('data', (data) => {
        const message = data.toString();
        
        if (!this.isInitialized) {
          initBuffer += message;
          if (initBuffer.includes('ADAM Python Backend Ready')) {
            console.log('ADAM backend initialization detected');
            resolve();
            return;
          }
        }

        this.handlePythonMessage(message);
      });

      this.pythonProcess.stderr?.on('data', (data) => {
        const message = data.toString().trim();
        if (message.includes('ADAM Python Backend Ready')) {
          console.log('✅ ADAM:', message);
          if (!this.isInitialized) {
            this.isInitialized = true;
            resolve();
          }
        } else {
          console.log('ADAM Status:', message);
        }
      });

      this.pythonProcess.on('error', (error) => {
        if (!this.isInitialized) {
          reject(error);
        } else {
          this.emit('error', error);
        }
      });

      this.pythonProcess.on('exit', (code) => {
        console.log(`ADAM Python process exited with code ${code}`);
        this.pythonProcess = null;
        this.isInitialized = false;
        this.emit('disconnected');
      });

      // Timeout for initialization
      setTimeout(() => {
        if (!this.isInitialized) {
          reject(new Error('Python backend initialization timeout'));
        }
      }, 30000);
    });
  }

  private handlePythonMessage(message: string): void {
    try {
      const lines = message.split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        if (line.startsWith('ADAM_RESPONSE:')) {
          const responseData = JSON.parse(line.substring('ADAM_RESPONSE:'.length));
          this.handleADAMResponse(responseData);
        } else if (line.startsWith('ADAM_ERROR:')) {
          const errorData = JSON.parse(line.substring('ADAM_ERROR:'.length));
          this.handleADAMError(errorData);
        }
      }
    } catch (error) {
      console.error('Error parsing Python message:', error);
    }
  }

  private handleADAMResponse(data: any): void {
    const requestId = data.requestId;
    const pendingRequest = this.requestQueue.get(requestId);
    
    if (pendingRequest) {
      clearTimeout(pendingRequest.timeout);
      this.requestQueue.delete(requestId);
      pendingRequest.resolve(data.response);
    }
  }

  private handleADAMError(data: any): void {
    const requestId = data.requestId;
    const pendingRequest = this.requestQueue.get(requestId);
    
    if (pendingRequest) {
      clearTimeout(pendingRequest.timeout);
      this.requestQueue.delete(requestId);
      pendingRequest.reject(data.error);
    }
  }

  async processQuery(request: ADAMRequest): Promise<ADAMResponse> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2)}`;
    
    return new Promise((resolve, reject) => {
      // Set up timeout
      const timeout = setTimeout(() => {
        this.requestQueue.delete(requestId);
        reject({
          error: 'Request timeout',
          code: 'TIMEOUT',
          retryable: true
        });
      }, 120000); // 2 minutes timeout

      // Store request in queue
      this.requestQueue.set(requestId, { resolve, reject, timeout });

      // Send request to Python
      const pythonRequest = {
        requestId,
        type: 'QUERY',
        data: request
      };

      if (this.pythonProcess?.stdin) {
        this.pythonProcess.stdin.write(JSON.stringify(pythonRequest) + '\n');
      } else {
        clearTimeout(timeout);
        this.requestQueue.delete(requestId);
        reject({
          error: 'Python process not available',
          code: 'NO_PROCESS',
          retryable: true
        });
      }
    });
  }

  async getCostSummary(): Promise<{
    dailyCost: number;
    monthlyCost: number;
    queryCount: number;
    modelUsage: Record<string, number>;
  }> {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const requestId = `cost_${Date.now()}`;
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.requestQueue.delete(requestId);
        reject({ error: 'Cost summary timeout', code: 'TIMEOUT', retryable: true });
      }, 10000);

      this.requestQueue.set(requestId, { 
        resolve: resolve as any, 
        reject: reject as any, 
        timeout 
      });

      const request = {
        requestId,
        type: 'COST_SUMMARY',
        data: {}
      };

      if (this.pythonProcess?.stdin) {
        this.pythonProcess.stdin.write(JSON.stringify(request) + '\n');
      } else {
        clearTimeout(timeout);
        this.requestQueue.delete(requestId);
        reject({ error: 'Python process not available', code: 'NO_PROCESS', retryable: true });
      }
    });
  }

  async shutdown(): Promise<void> {
    if (this.pythonProcess) {
      // Clear all pending requests
      for (const [requestId, request] of Array.from(this.requestQueue.entries())) {
        clearTimeout(request.timeout);
        request.reject({
          error: 'Service shutting down',
          code: 'SHUTDOWN',
          retryable: false
        });
      }
      this.requestQueue.clear();

      // Gracefully close Python process
      this.pythonProcess.stdin?.end();
      
      return new Promise((resolve) => {
        if (this.pythonProcess) {
          this.pythonProcess.on('exit', () => resolve());
          setTimeout(() => {
            this.pythonProcess?.kill('SIGTERM');
            setTimeout(() => {
              this.pythonProcess?.kill('SIGKILL');
              resolve();
            }, 5000);
          }, 5000);
        } else {
          resolve();
        }
      });
    }
  }
}

// Singleton instance
let adamBridge: ADAMPythonBridge | null = null;

export function getADAMBridge(): ADAMPythonBridge {
  if (!adamBridge) {
    adamBridge = new ADAMPythonBridge();
  }
  return adamBridge;
}

export { ADAMPythonBridge };