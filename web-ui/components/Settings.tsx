"use client";

import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Settings as SettingsIcon, 
  Download, 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  Eye, 
  EyeOff, 
  Server,
  Database,
  Key,
  Cpu,
  HardDrive,
  Clock,
  RefreshCw
} from "lucide-react";
import { api, ModelStatus, ModelInfo, HFTokenStatus, SystemStatus } from "@/lib/api";
import { cn } from "@/lib/utils";

interface SettingsProps {
  systemStatus: SystemStatus | null;
}

export default function Settings({ systemStatus }: SettingsProps) {
  const [activeSection, setActiveSection] = useState("models");
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null);
  const [hfTokenStatus, setHFTokenStatus] = useState<HFTokenStatus | null>(null);
  const [hfToken, setHFToken] = useState("");
  const [showToken, setShowToken] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [downloadingModels, setDownloadingModels] = useState<Set<string>>(new Set());

  useEffect(() => {
    loadModelStatus();
    loadHFTokenStatus();
  }, []);

  useEffect(() => {
    // Handle WebSocket messages for model download progress
    if (api) {
      const originalConnectWebSocket = api.connectWebSocket.bind(api);
      // Store the original onMessage handler to avoid breaking existing functionality
      const handleWebSocketMessage = (data: unknown) => {
        if (typeof data === 'object' && data !== null && 'type' in data) {
          const msg = data as Record<string, unknown>;
          
          if (msg.type === 'model_download_started') {
            setDownloadingModels(prev => new Set(prev).add(msg.model_id as string));
          } else if (msg.type === 'model_download_completed' || msg.type === 'model_download_error') {
            setDownloadingModels(prev => {
              const newSet = new Set(prev);
              newSet.delete(msg.model_id as string);
              return newSet;
            });
            loadModelStatus(); // Refresh model status
          }
        }
      };
      
      // Connect with our message handler
      api.connectWebSocket(handleWebSocketMessage);
    }
  }, []);

  const loadModelStatus = async () => {
    try {
      const status = await api.getModelStatus();
      setModelStatus(status);
    } catch (error) {
      console.error('Failed to load model status:', error);
    }
  };

  const loadHFTokenStatus = async () => {
    try {
      const status = await api.getHFTokenStatus();
      setHFTokenStatus(status);
    } catch (error) {
      console.error('Failed to load HF token status:', error);
    }
  };

  const handleHFTokenSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!hfToken.trim()) return;

    setIsSubmitting(true);
    try {
      await api.setHFToken(hfToken.trim());
      setMessage({ type: 'success', text: 'HuggingFace token saved successfully!' });
      
      // Store in localStorage as well (like agenticinsights.com pattern)
      localStorage.setItem('hf_token', hfToken.trim());
      
      setHFToken("");
      await loadHFTokenStatus();
    } catch (error) {
      setMessage({ 
        type: 'error', 
        text: `Failed to save token: ${error instanceof Error ? error.message : 'Unknown error'}` 
      });
    } finally {
      setIsSubmitting(false);
    }

    // Clear message after 5 seconds
    setTimeout(() => setMessage(null), 5000);
  };

  const handleModelDownload = async (modelId: string) => {
    try {
      setDownloadingModels(prev => new Set(prev).add(modelId));
      await api.downloadModel(modelId);
      setMessage({ type: 'success', text: `Started downloading ${modelId}` });
    } catch (error) {
      setDownloadingModels(prev => {
        const newSet = new Set(prev);
        newSet.delete(modelId);
        return newSet;
      });
      setMessage({ 
        type: 'error', 
        text: `Failed to start download: ${error instanceof Error ? error.message : 'Unknown error'}` 
      });
    }

    // Clear message after 5 seconds
    setTimeout(() => setMessage(null), 5000);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getModelStatusColor = (status: ModelInfo['status']) => {
    switch (status) {
      case 'ready': return 'text-green-500';
      case 'downloading': return 'text-yellow-500';
      case 'partial': return 'text-orange-500';
      case 'not_downloaded': return 'text-gray-400';
      default: return 'text-gray-400';
    }
  };

  const getModelStatusIcon = (model: ModelInfo) => {
    const isDownloading = downloadingModels.has(model.id);
    
    if (isDownloading) {
      return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />;
    }
    
    switch (model.status) {
      case 'ready': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'downloading': return <Loader2 className="w-4 h-4 text-yellow-500 animate-spin" />;
      case 'partial': return <AlertCircle className="w-4 h-4 text-orange-500" />;
      case 'not_downloaded': return <Download className="w-4 h-4 text-gray-400" />;
      default: return <AlertCircle className="w-4 h-4 text-gray-400" />;
    }
  };

  const sections = [
    { id: "models", label: "Models", icon: Database },
    { id: "auth", label: "Authentication", icon: Key },
    { id: "system", label: "System", icon: Server },
  ];

  return (
    <div className="h-full flex flex-col lg:flex-row">
      {/* Settings Sidebar */}
      <div className="lg:w-64 border-b lg:border-b-0 lg:border-r border-border bg-card">
        <div className="p-4">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <SettingsIcon className="w-5 h-5 text-primary" />
            Settings
          </h2>
          <nav className="space-y-1">
            {sections.map((section) => {
              const Icon = section.icon;
              return (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={cn(
                    "w-full flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-colors",
                    "hover:bg-background",
                    activeSection === section.id
                      ? "bg-background text-foreground font-medium"
                      : "text-muted-foreground"
                  )}
                >
                  <Icon className="w-4 h-4" />
                  {section.label}
                </button>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Settings Content */}
      <div className="flex-1 overflow-y-auto">
        <AnimatePresence mode="wait">
          {/* Models Section */}
          {activeSection === "models" && (
            <motion.div
              key="models"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 sm:p-6 lg:p-8"
            >
              <div className="max-w-4xl">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="text-xl font-semibold">Model Management</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Download and manage AI models for image generation
                    </p>
                  </div>
                  <button
                    onClick={loadModelStatus}
                    className="p-2 hover:bg-background rounded-md transition-colors"
                    title="Refresh model status"
                  >
                    <RefreshCw className="w-4 h-4 text-muted-foreground" />
                  </button>
                </div>

                {modelStatus && (
                  <div className="mb-6 p-4 bg-muted/50 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <HardDrive className="w-4 h-4 text-muted-foreground" />
                      <span className="text-sm font-medium">Cache Directory</span>
                    </div>
                    <code className="text-xs text-muted-foreground break-all">
                      {modelStatus.cache_dir}
                    </code>
                  </div>
                )}

                <div className="space-y-4">
                  {modelStatus?.models.map((model) => {
                    const isDownloading = downloadingModels.has(model.id);
                    const canDownload = model.status === 'not_downloaded' || model.status === 'partial';
                    
                    return (
                      <div
                        key={model.id}
                        className="border border-border rounded-lg p-4 hover:bg-muted/30 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            {getModelStatusIcon(model)}
                            <div>
                              <h4 className="font-medium">{model.name}</h4>
                              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                                <span className="capitalize">{model.type.replace('-', ' ')}</span>
                                <span>•</span>
                                <span className={getModelStatusColor(model.status)}>
                                  {model.status.replace('_', ' ').toUpperCase()}
                                </span>
                                {model.size > 0 && (
                                  <>
                                    <span>•</span>
                                    <span>{formatFileSize(model.size)}</span>
                                  </>
                                )}
                              </div>
                              {model.incomplete_files > 0 && (
                                <div className="text-xs text-orange-500 mt-1">
                                  {model.incomplete_files} files downloading...
                                </div>
                              )}
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-2">
                            {canDownload && !isDownloading && (
                              <button
                                onClick={() => handleModelDownload(model.id)}
                                className="px-3 py-1.5 bg-primary text-primary-foreground text-xs rounded-md hover:opacity-90 transition-opacity"
                              >
                                Download
                              </button>
                            )}
                            {isDownloading && (
                              <div className="flex items-center gap-2 text-xs text-yellow-600">
                                <Loader2 className="w-3 h-3 animate-spin" />
                                Downloading...
                              </div>
                            )}
                            {model.status === 'ready' && (
                              <div className="flex items-center gap-2 text-xs text-green-600">
                                <CheckCircle className="w-3 h-3" />
                                Ready
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {/* Authentication Section */}
          {activeSection === "auth" && (
            <motion.div
              key="auth"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 sm:p-6 lg:p-8"
            >
              <div className="max-w-2xl">
                <h3 className="text-xl font-semibold mb-6">Authentication</h3>
                
                {/* HF Token Section */}
                <div className="space-y-6">
                  <div className="border border-border rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-4">
                      <Key className="w-5 h-5 text-primary" />
                      <h4 className="text-lg font-medium">HuggingFace Token</h4>
                      {hfTokenStatus?.configured && (
                        <div className="flex items-center gap-1 text-xs text-green-600">
                          <CheckCircle className="w-3 h-3" />
                          Configured ({hfTokenStatus.source})
                        </div>
                      )}
                    </div>
                    
                    <p className="text-sm text-muted-foreground mb-4">
                      Required for downloading private models and avoiding rate limits. 
                      Get your token from{" "}
                      <a 
                        href="https://huggingface.co/settings/tokens" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="text-primary hover:underline"
                      >
                        HuggingFace Settings
                      </a>
                    </p>

                    <form onSubmit={handleHFTokenSubmit} className="space-y-4">
                      <div className="relative">
                        <input
                          type={showToken ? "text" : "password"}
                          value={hfToken}
                          onChange={(e) => setHFToken(e.target.value)}
                          placeholder="hf_..."
                          className="w-full px-3 py-2 pr-10 bg-background border border-input rounded-md text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                        <button
                          type="button"
                          onClick={() => setShowToken(!showToken)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        >
                          {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>
                      
                      <button
                        type="submit"
                        disabled={!hfToken.trim() || isSubmitting}
                        className="px-4 py-2 bg-primary text-primary-foreground rounded-md hover:opacity-90 disabled:opacity-50 transition-opacity text-sm font-medium"
                      >
                        {isSubmitting ? (
                          <div className="flex items-center gap-2">
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Saving...
                          </div>
                        ) : (
                          "Save Token"
                        )}
                      </button>
                    </form>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* System Section */}
          {activeSection === "system" && (
            <motion.div
              key="system"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 sm:p-6 lg:p-8"
            >
              <div className="max-w-2xl">
                <h3 className="text-xl font-semibold mb-6">System Information</h3>
                
                {systemStatus && (
                  <div className="space-y-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="border border-border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Server className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm font-medium">API Status</span>
                        </div>
                        <div className={cn(
                          "text-lg font-semibold",
                          systemStatus.status === 'ready' ? 'text-green-500' : 'text-orange-500'
                        )}>
                          {systemStatus.status === 'ready' ? 'Ready' : 'Not Ready'}
                        </div>
                      </div>

                      <div className="border border-border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Cpu className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm font-medium">GPU</span>
                        </div>
                        <div className={cn(
                          "text-lg font-semibold",
                          systemStatus.gpu_available ? 'text-green-500' : 'text-orange-500'
                        )}>
                          {systemStatus.gpu_available ? 'Available' : 'Not Available'}
                        </div>
                      </div>

                      <div className="border border-border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Database className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Backend</span>
                        </div>
                        <div className="text-lg font-semibold text-foreground capitalize">
                          {systemStatus.backend}
                        </div>
                      </div>

                      <div className="border border-border rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <SettingsIcon className="w-4 h-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Plugins</span>
                        </div>
                        <div className="text-lg font-semibold text-foreground">
                          {systemStatus.active_plugins.length} Active
                        </div>
                      </div>
                    </div>

                    {systemStatus.active_plugins.length > 0 && (
                      <div className="border border-border rounded-lg p-4">
                        <h4 className="text-sm font-medium mb-3">Active Plugins</h4>
                        <div className="flex flex-wrap gap-2">
                          {systemStatus.active_plugins.map((plugin) => (
                            <span
                              key={plugin}
                              className="px-2 py-1 bg-primary/10 text-primary text-xs rounded-md"
                            >
                              {plugin}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Status Message */}
      <AnimatePresence>
        {message && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed bottom-4 right-4 z-50"
          >
            <div className={cn(
              "px-4 py-3 rounded-lg shadow-lg text-sm font-medium",
              message.type === 'success' 
                ? "bg-green-500 text-white" 
                : "bg-red-500 text-white"
            )}>
              {message.text}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}