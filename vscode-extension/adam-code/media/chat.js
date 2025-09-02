(function() {
    const vscode = acquireVsCodeApi();
    
    // DOM elements
    const messagesContainer = document.getElementById('messages');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const clearBtn = document.getElementById('clearBtn');
    const exportBtn = document.getElementById('exportBtn');
    const styleSelect = document.getElementById('styleSelect');
    const typingIndicator = document.getElementById('typingIndicator');

    // Load saved style preference
    const savedStyle = localStorage.getItem('adam-response-style') || 'normal';
    if (styleSelect) {
        styleSelect.value = savedStyle;
    }

    // Handle style change
    if (styleSelect) {
        styleSelect.addEventListener('change', (e) => {
            const style = e.target.value;
            localStorage.setItem('adam-response-style', style);
            vscode.postMessage({
                type: 'changeStyle',
                style: style
            });
        });
    }

    // Handle send message
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        vscode.postMessage({
            type: 'sendMessage',
            message: message
        });

        messageInput.value = '';
        messageInput.style.height = 'auto';
    }

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
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // Handle clear
    clearBtn.addEventListener('click', () => {
        if (confirm('Clear all messages?')) {
            vscode.postMessage({ type: 'clear' });
        }
    });

    // Handle export
    exportBtn.addEventListener('click', () => {
        vscode.postMessage({ type: 'export' });
    });

    // Handle messages from extension
    window.addEventListener('message', event => {
        const message = event.data;
        
        switch (message.type) {
            case 'messages':
                displayMessages(message.messages);
                break;
            case 'typing':
                typingIndicator.style.display = message.isTyping ? 'block' : 'none';
                break;
        }
    });

    function displayMessages(messages) {
        messagesContainer.innerHTML = '';
        
        messages.forEach(msg => {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${msg.role}`;
            
            // Format message content with markdown support
            let content = msg.content;
            
            // Convert markdown code blocks to HTML
            content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre><code class="language-${lang || 'plaintext'}">${escapeHtml(code.trim())}</code></pre>`;
            });
            
            // Convert inline code
            content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
            
            // Convert bold
            content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // Convert italic
            content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
            
            // Convert line breaks
            content = content.replace(/\n/g, '<br>');
            
            // Build header with model info
            let headerText = msg.role === 'user' ? '👤 You' : '🧠 ADAM';
            if (msg.role === 'assistant' && msg.model) {
                headerText += ` <span style="opacity: 0.6; font-size: 11px;">(${msg.model})</span>`;
            }
            
            messageDiv.innerHTML = `
                <div class="message-header">${headerText}</div>
                <div class="message-content">${content}</div>
                ${msg.cost ? `<div class="message-cost">Cost: $${msg.cost.toFixed(4)}</div>` : ''}
            `;
            
            messagesContainer.appendChild(messageDiv);
        });
        
        // Scroll to bottom
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, m => map[m]);
    }
})();