(function() {
    const vscode = acquireVsCodeApi();
    
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const exportBtn = document.getElementById('exportBtn');
    const typingIndicator = document.getElementById('typingIndicator');
    
    let messages = [];
    
    // Send message
    function sendMessage() {
        const content = messageInput.value.trim();
        if (!content) return;
        
        vscode.postMessage({
            type: 'sendMessage',
            message: content
        });
        
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
    
    // Render messages
    function renderMessages() {
        messagesContainer.innerHTML = '';
        
        messages.forEach((msg, index) => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.innerHTML = formatMessage(msg.content);
            
            messageDiv.appendChild(contentDiv);
            
            if (msg.model || msg.cost) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                const metaParts = [];
                if (msg.model) metaParts.push(`Model: ${msg.model}`);
                if (msg.cost) metaParts.push(`Cost: $${msg.cost.toFixed(4)}`);
                metaDiv.textContent = metaParts.join(' • ');
                messageDiv.appendChild(metaDiv);
            }
            
            messagesContainer.appendChild(messageDiv);
        });
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    // Format message with markdown support
    function formatMessage(content) {
        // Basic markdown parsing
        let formatted = content
            // Code blocks
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre><code class="language-${lang || 'plaintext'}">${escapeHtml(code.trim())}</code></pre>`;
            })
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Bold
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*([^*]+)\*/g, '<em>$1</em>')
            // Headers
            .replace(/^### (.+)$/gm, '<h3>$1</h3>')
            .replace(/^## (.+)$/gm, '<h2>$1</h2>')
            .replace(/^# (.+)$/gm, '<h1>$1</h1>')
            // Lists
            .replace(/^\* (.+)$/gm, '<li>$1</li>')
            .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
            // Line breaks
            .replace(/\n/g, '<br>');
        
        // Wrap consecutive list items
        formatted = formatted.replace(/(<li>.*<\/li>(<br>)?)+/g, (match) => {
            return `<ul>${match.replace(/<br>/g, '')}</ul>`;
        });
        
        return formatted;
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Event listeners
    sendBtn.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', () => {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
    });
    
    clearBtn.addEventListener('click', () => {
        if (confirm('Clear all messages?')) {
            vscode.postMessage({ type: 'clear' });
        }
    });
    
    exportBtn.addEventListener('click', () => {
        vscode.postMessage({ type: 'export' });
    });
    
    // Handle messages from extension
    window.addEventListener('message', event => {
        const message = event.data;
        
        switch (message.type) {
            case 'messages':
                messages = message.messages;
                renderMessages();
                break;
            case 'typing':
                typingIndicator.style.display = message.isTyping ? 'flex' : 'none';
                if (message.isTyping) {
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                }
                break;
        }
    });
    
    // Focus input on load
    messageInput.focus();
    
    // Notify extension that webview is ready
    vscode.postMessage({ type: 'ready' });
})();