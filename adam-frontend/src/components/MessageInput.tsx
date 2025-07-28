import { useState, useRef } from 'react';
import type { KeyboardEvent } from 'react';

interface MessageInputProps {
  onSendMessage: (content: string, imageData?: string) => void;
  disabled?: boolean;
}

export function MessageInput({ onSendMessage, disabled }: MessageInputProps) {
  const [content, setContent] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e?: React.FormEvent) => {
    e?.preventDefault();
    
    if (!content.trim() && !selectedFile) return;
    
    let imageData: string | undefined;
    if (selectedFile) {
      imageData = await fileToBase64(selectedFile);
    }
    
    onSendMessage(content.trim(), imageData);
    setContent('');
    clearFile();
    
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setContent(e.target.value);
    
    // Auto-resize textarea
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      const preview = await fileToBase64(file);
      setImagePreview(preview);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = reject;
    });
  };

  return (
    <div className="bg-gray-800 border-t border-gray-700 p-4">
      <form onSubmit={handleSubmit}>
        <div className="flex space-x-4">
          <div className="flex-1">
            <textarea
              ref={textareaRef}
              value={content}
              onChange={handleTextareaChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Shift+Enter for new line)"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
              rows={3}
              disabled={disabled}
            />
            
            {/* File Preview */}
            {imagePreview && (
              <div className="mt-2">
                <div className="inline-flex items-center bg-gray-700 rounded px-3 py-1">
                  <img
                    src={imagePreview}
                    alt="Preview"
                    className="h-16 w-16 object-cover rounded mr-2"
                  />
                  <span className="text-sm text-white">{selectedFile?.name}</span>
                  <button
                    type="button"
                    onClick={clearFile}
                    className="ml-2 text-gray-400 hover:text-white"
                  >
                    <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            )}
          </div>
          
          <div className="flex flex-col space-y-2">
            <button
              type="submit"
              disabled={disabled || (!content.trim() && !selectedFile)}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center justify-center space-x-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
              <span>Send</span>
            </button>
            
            <label className="text-center cursor-pointer text-gray-400 hover:text-gray-300 transition-colors">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                className="hidden"
                disabled={disabled}
              />
              <svg className="w-5 h-5 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <span className="text-xs">Image</span>
            </label>
          </div>
        </div>
      </form>
    </div>
  );
}