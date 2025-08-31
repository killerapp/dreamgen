const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface GenerateRequest {
  prompt?: string;
  use_mock?: boolean;
  enable_plugins?: boolean;
  seed?: number;
}

export interface GenerateResponse {
  id: string;
  prompt: string;
  image_path: string;
  metadata: {
    backend: string;
    plugins_used: string[];
    seed?: number;
  };
  created_at: string;
}

export interface PluginInfo {
  name: string;
  enabled: boolean;
  description: string;
}

export interface SystemStatus {
  status: string;
  backend: string;
  plugins_enabled: boolean;
  active_plugins: string[];
  gpu_available: boolean;
  ollama_available: boolean;
}

export class ImageGenAPI {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private intentionalClose: boolean = false;

  constructor(baseUrl: string = API_BASE) {
    this.baseUrl = baseUrl;
  }

  async getStatus(): Promise<SystemStatus> {
    const response = await fetch(`${this.baseUrl}/api/status`);
    if (!response.ok) throw new Error('Failed to get status');
    return response.json();
  }

  async getPlugins(): Promise<PluginInfo[]> {
    const response = await fetch(`${this.baseUrl}/api/plugins`);
    if (!response.ok) throw new Error('Failed to get plugins');
    return response.json();
  }

  async togglePlugin(pluginName: string, enabled: boolean): Promise<PluginInfo> {
    const response = await fetch(`${this.baseUrl}/api/plugins/${pluginName}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ enabled }),
    });
    if (!response.ok) throw new Error('Failed to toggle plugin');
    return response.json();
  }

  async generate(request: GenerateRequest): Promise<GenerateResponse> {
    const response = await fetch(`${this.baseUrl}/api/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    if (!response.ok) throw new Error('Failed to generate image');
    return response.json();
  }

  async getGallery(limit: number = 50, offset: number = 0) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
    
    try {
      const response = await fetch(
        `${this.baseUrl}/api/gallery?limit=${limit}&offset=${offset}`,
        { signal: controller.signal }
      );
      clearTimeout(timeoutId);
      
      if (!response.ok) throw new Error('Failed to get gallery');
      return response.json();
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof Error && error.name === 'AbortError') {
        throw new Error('Gallery request timed out');
      }
      throw error;
    }
  }

  async deleteImage(imagePath: string) {
    const response = await fetch(`${this.baseUrl}/api/gallery/${encodeURIComponent(imagePath)}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error('Failed to delete image');
    return response.json();
  }

  connectWebSocket(onMessage: (data: unknown) => void): void {
    const wsUrl = this.baseUrl.replace('http://', 'ws://').replace('https://', 'wss://');
    
    try {
      this.ws = new WebSocket(`${wsUrl}/ws`);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.warn('Failed to parse WebSocket message:', error instanceof Error ? error.message : 'Unknown error');
        }
      };

      this.ws.onerror = (event) => {
        // WebSocket error events don't contain much useful information
        // Just log a simple message instead of trying to log the event object
        console.warn('WebSocket connection error occurred');
      };

      this.ws.onclose = (event) => {
        console.log('WebSocket disconnected');
        // Only reconnect if it wasn't an intentional closure
        if (!this.intentionalClose && event.code !== 1000) {
          // Attempt reconnection after 3 seconds
          setTimeout(() => this.connectWebSocket(onMessage), 3000);
        }
        this.intentionalClose = false;
      };
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error instanceof Error ? error.message : 'Unknown error');
      // Retry connection after 3 seconds
      setTimeout(() => this.connectWebSocket(onMessage), 3000);
    }
  }

  disconnectWebSocket(): void {
    if (this.ws) {
      this.intentionalClose = true;
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  sendWebSocketMessage(message: unknown): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }
}

export const api = new ImageGenAPI();