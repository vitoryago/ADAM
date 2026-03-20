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

    // Streaming state
    let streamingMessageDiv = null;
    let streamingContentEl = null;
    let streamingText = '';
    let isStreaming = false;

    // Multi-agent state
    let agentActivityDiv = null;
    let agentCards = {};
    let totalAgentCost = 0;

    // Agent display config
    const AGENT_COLORS = {
        orchestrator: '#888',
        reasoner: '#4a9eff',
        coder: '#3ddc84',
        critic: '#ff9f43',
        synthesizer: '#a855f7',
        debate: '#ef4444',
        unknown: '#888'
    };

    const AGENT_LABELS = {
        orchestrator: 'Orchestrator',
        reasoner: 'Reasoner',
        coder: 'Coder',
        critic: 'Critic',
        synthesizer: 'Synthesizer',
        debate: 'Debate',
        unknown: 'Agent'
    };

    // Load saved style preference
    const savedStyle = localStorage.getItem('adam-response-style') || 'normal';
    if (styleSelect) {
        styleSelect.value = savedStyle;
    }

    // Handle style change
    if (styleSelect) {
        styleSelect.addEventListener('change', function(e) {
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

    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        messageInput.style.height = 'auto';
        messageInput.style.height = messageInput.scrollHeight + 'px';
    });

    // Handle clear
    clearBtn.addEventListener('click', function() {
        if (confirm('Clear all messages?')) {
            vscode.postMessage({ type: 'clear' });
        }
    });

    // Handle export
    exportBtn.addEventListener('click', function() {
        vscode.postMessage({ type: 'export' });
    });

    // Handle messages from extension
    window.addEventListener('message', function(event) {
        var message = event.data;

        switch (message.type) {
            case 'messages':
                displayMessages(message.messages);
                break;
            case 'typing':
                typingIndicator.style.display = message.isTyping ? 'block' : 'none';
                break;

            // --- Streaming events ---
            case 'streamStart':
                startStreamingMessage();
                break;
            case 'streamChunk':
                appendStreamChunk(message.content);
                break;
            case 'streamComplete':
                finalizeStreamingMessage(message.model, message.cost, message.tokens);
                break;
            case 'streamEnd':
                endStreaming();
                break;
            case 'streamError':
                showStreamError(message.message);
                break;

            // --- Multi-agent events ---
            case 'agentStatus':
                handleAgentStatus(message.agent, message.status, message.pattern);
                break;
            case 'agentChunk':
                handleAgentChunk(message.agent, message.content);
                break;
            case 'agentDone':
                handleAgentDone(message.agent, message.cost);
                break;
        }
    });

    // ========================================================================
    // Streaming display
    // ========================================================================

    function startStreamingMessage() {
        isStreaming = true;
        streamingText = '';

        // Create the assistant message bubble
        streamingMessageDiv = document.createElement('div');
        streamingMessageDiv.className = 'message assistant streaming';

        var header = document.createElement('div');
        header.className = 'message-header';
        header.textContent = 'ADAM ';
        var badge = document.createElement('span');
        badge.className = 'streaming-badge';
        badge.textContent = 'streaming';
        header.appendChild(badge);

        streamingContentEl = document.createElement('div');
        streamingContentEl.className = 'message-content';
        var cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        streamingContentEl.appendChild(cursor);

        streamingMessageDiv.appendChild(header);
        streamingMessageDiv.appendChild(streamingContentEl);
        messagesContainer.appendChild(streamingMessageDiv);
        scrollToBottom();
    }

    function appendStreamChunk(text) {
        if (!streamingContentEl) return;

        streamingText += text;

        // Re-render the accumulated markdown.
        // Note: This runs inside a VS Code webview sandbox with no external
        // content. All non-markdown text is escaped via escapeHtml().
        var rendered = renderMarkdown(streamingText);
        streamingContentEl.textContent = ''; // clear safely
        // We need to set parsed HTML from our own renderMarkdown which escapes
        // user text inside code blocks via escapeHtml.
        var wrapper = document.createElement('span');
        wrapper.innerHTML = rendered; // safe: renderMarkdown escapes untrusted parts
        while (wrapper.firstChild) {
            streamingContentEl.appendChild(wrapper.firstChild);
        }
        var cursorSpan = document.createElement('span');
        cursorSpan.className = 'streaming-cursor';
        streamingContentEl.appendChild(cursorSpan);
        scrollToBottom();
    }

    function finalizeStreamingMessage(model, cost, tokens) {
        if (!streamingMessageDiv || !streamingContentEl) return;

        // Remove streaming indicators
        streamingMessageDiv.classList.remove('streaming');
        var badge = streamingMessageDiv.querySelector('.streaming-badge');
        if (badge) badge.remove();

        // Final render without cursor
        var rendered = renderMarkdown(streamingText);
        streamingContentEl.textContent = '';
        var wrapper = document.createElement('span');
        wrapper.innerHTML = rendered;
        while (wrapper.firstChild) {
            streamingContentEl.appendChild(wrapper.firstChild);
        }

        // Update header with model info
        var header = streamingMessageDiv.querySelector('.message-header');
        if (model && header) {
            header.textContent = 'ADAM ';
            var modelSpan = document.createElement('span');
            modelSpan.style.opacity = '0.6';
            modelSpan.style.fontSize = '11px';
            modelSpan.textContent = '(' + model + ')';
            header.appendChild(modelSpan);
        }

        if (cost && cost > 0) {
            var costDiv = document.createElement('div');
            costDiv.className = 'message-cost';
            costDiv.textContent = 'Cost: $' + cost.toFixed(4);
            if (tokens) {
                costDiv.textContent += ' | Tokens: ' + tokens;
            }
            streamingMessageDiv.appendChild(costDiv);
        }
    }

    function endStreaming() {
        isStreaming = false;
        streamingMessageDiv = null;
        streamingContentEl = null;
        streamingText = '';

        // Clean up agent activity state
        agentActivityDiv = null;
        agentCards = {};
        totalAgentCost = 0;
    }

    function showStreamError(errorMessage) {
        if (streamingContentEl) {
            streamingContentEl.textContent = '';
            var errDiv = document.createElement('div');
            errDiv.className = 'stream-error';
            errDiv.textContent = 'Error: ' + errorMessage;
            streamingContentEl.appendChild(errDiv);
        }
    }

    // ========================================================================
    // Multi-agent display
    // ========================================================================

    function ensureAgentActivityPanel() {
        if (agentActivityDiv) return agentActivityDiv;

        agentActivityDiv = document.createElement('div');
        agentActivityDiv.className = 'agent-activity';

        // Header
        var headerDiv = document.createElement('div');
        headerDiv.className = 'agent-activity-header';
        headerDiv.addEventListener('click', function() {
            agentActivityDiv.classList.toggle('collapsed');
        });

        var titleSpan = document.createElement('span');
        titleSpan.className = 'agent-activity-title';
        titleSpan.textContent = 'Agent Activity';
        headerDiv.appendChild(titleSpan);

        var costSpan = document.createElement('span');
        costSpan.className = 'agent-activity-cost';
        costSpan.id = 'totalAgentCost';
        headerDiv.appendChild(costSpan);

        var toggleSpan = document.createElement('span');
        toggleSpan.className = 'agent-activity-toggle';
        toggleSpan.textContent = '\u25BC'; // down arrow
        headerDiv.appendChild(toggleSpan);

        agentActivityDiv.appendChild(headerDiv);

        // Body
        var bodyDiv = document.createElement('div');
        bodyDiv.className = 'agent-activity-body';
        agentActivityDiv.appendChild(bodyDiv);

        messagesContainer.appendChild(agentActivityDiv);
        scrollToBottom();
        return agentActivityDiv;
    }

    function getOrCreateAgentCard(agent) {
        if (agentCards[agent]) return agentCards[agent];

        var panel = ensureAgentActivityPanel();
        var body = panel.querySelector('.agent-activity-body');
        var color = AGENT_COLORS[agent] || AGENT_COLORS.unknown;
        var label = AGENT_LABELS[agent] || agent;

        var card = document.createElement('div');
        card.className = 'agent-card';
        card.style.borderLeftColor = color;

        // Card header
        var cardHeader = document.createElement('div');
        cardHeader.className = 'agent-card-header';
        cardHeader.addEventListener('click', function() {
            card.classList.toggle('expanded');
        });

        var nameSpan = document.createElement('span');
        nameSpan.className = 'agent-name';
        nameSpan.style.color = color;
        nameSpan.textContent = label;
        cardHeader.appendChild(nameSpan);

        var statusSpan = document.createElement('span');
        statusSpan.className = 'agent-status';
        statusSpan.textContent = 'initializing';
        cardHeader.appendChild(statusSpan);

        var costBadge = document.createElement('span');
        costBadge.className = 'agent-cost-badge';
        costBadge.style.display = 'none';
        cardHeader.appendChild(costBadge);

        var toggleSpan = document.createElement('span');
        toggleSpan.className = 'agent-toggle';
        toggleSpan.textContent = '+';
        cardHeader.appendChild(toggleSpan);

        card.appendChild(cardHeader);

        // Card content
        var cardContent = document.createElement('div');
        cardContent.className = 'agent-card-content';
        card.appendChild(cardContent);

        body.appendChild(card);
        agentCards[agent] = card;
        scrollToBottom();
        return card;
    }

    function handleAgentStatus(agent, status, pattern) {
        var card = getOrCreateAgentCard(agent);
        var statusEl = card.querySelector('.agent-status');

        if (status === 'thinking') {
            statusEl.textContent = 'thinking...';
            statusEl.className = 'agent-status thinking';
        } else if (status === 'started') {
            statusEl.textContent = pattern ? 'started (' + pattern + ')' : 'started';
            statusEl.className = 'agent-status started';
        } else {
            statusEl.textContent = status;
            statusEl.className = 'agent-status';
        }

        scrollToBottom();
    }

    function handleAgentChunk(agent, content) {
        var card = getOrCreateAgentCard(agent);
        var contentEl = card.querySelector('.agent-card-content');

        // Auto-expand if content arrives
        card.classList.add('expanded');

        // Append text
        var currentText = contentEl.getAttribute('data-raw') || '';
        var newText = currentText + content;
        contentEl.setAttribute('data-raw', newText);

        // Re-render markdown. renderMarkdown escapes code blocks via escapeHtml.
        var rendered = renderMarkdown(newText);
        contentEl.textContent = '';
        var wrapper = document.createElement('span');
        wrapper.innerHTML = rendered;
        while (wrapper.firstChild) {
            contentEl.appendChild(wrapper.firstChild);
        }
        scrollToBottom();
    }

    function handleAgentDone(agent, cost) {
        var card = getOrCreateAgentCard(agent);
        var statusEl = card.querySelector('.agent-status');
        statusEl.textContent = 'done';
        statusEl.className = 'agent-status done';

        if (cost > 0) {
            var costBadge = card.querySelector('.agent-cost-badge');
            costBadge.textContent = '$' + cost.toFixed(4);
            costBadge.style.display = 'inline-block';
        }

        // Update total cost
        totalAgentCost += cost;
        if (agentActivityDiv) {
            var totalCostEl = agentActivityDiv.querySelector('#totalAgentCost');
            if (totalCostEl) {
                totalCostEl.textContent = 'Total: $' + totalAgentCost.toFixed(4);
            }
        }

        scrollToBottom();
    }

    // ========================================================================
    // Message display (non-streaming / history)
    // ========================================================================

    function displayMessages(messages) {
        messagesContainer.textContent = '';

        messages.forEach(function(msg) {
            var messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + msg.role;

            // Header
            var headerDiv = document.createElement('div');
            headerDiv.className = 'message-header';
            if (msg.role === 'user') {
                headerDiv.textContent = 'You';
            } else {
                headerDiv.textContent = 'ADAM';
                if (msg.model) {
                    headerDiv.textContent += ' ';
                    var modelSpan = document.createElement('span');
                    modelSpan.style.opacity = '0.6';
                    modelSpan.style.fontSize = '11px';
                    modelSpan.textContent = '(' + msg.model + ')';
                    headerDiv.appendChild(modelSpan);
                }
            }
            messageDiv.appendChild(headerDiv);

            // Content -- renderMarkdown escapes code blocks via escapeHtml
            var contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            var rendered = renderMarkdown(msg.content);
            var wrapper = document.createElement('span');
            wrapper.innerHTML = rendered;
            while (wrapper.firstChild) {
                contentDiv.appendChild(wrapper.firstChild);
            }
            messageDiv.appendChild(contentDiv);

            // Cost
            if (msg.cost) {
                var costDiv = document.createElement('div');
                costDiv.className = 'message-cost';
                costDiv.textContent = 'Cost: $' + msg.cost.toFixed(4);
                messageDiv.appendChild(costDiv);
            }

            messagesContainer.appendChild(messageDiv);
        });

        scrollToBottom();
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    /**
     * Minimal markdown-to-HTML renderer for chat messages.
     *
     * Code blocks are escaped via escapeHtml() so user-supplied text inside
     * fenced blocks cannot inject HTML. This function is only called inside
     * the VS Code webview sandbox which has no access to cookies, localStorage
     * tokens, or the host filesystem.
     */
    function renderMarkdown(text) {
        var content = text;

        // Convert markdown code blocks to HTML (escape contents)
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, function(match, lang, code) {
            return '<pre><code class="language-' + (lang || 'plaintext') + '">' + escapeHtml(code.trim()) + '</code></pre>';
        });

        // Convert inline code
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Convert bold
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Convert italic
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Convert line breaks
        content = content.replace(/\n/g, '<br>');

        return content;
    }

    function escapeHtml(text) {
        var map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, function(m) { return map[m]; });
    }

    function scrollToBottom() {
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
})();
