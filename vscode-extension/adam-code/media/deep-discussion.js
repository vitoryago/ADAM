(function() {
    var vscode = acquireVsCodeApi();

    // ========================================================================
    // State
    // ========================================================================

    var mode = 'config';
    var agents = [];          // { role, model, status, content, cost, tokens, expanded }
    var completedSteps = [];
    var currentStep = '';
    var totalCost = 0;
    var sessionResult = '';
    var historySessions = [];
    var currentPattern = 'peer_review';
    var advancedExpanded = false;

    // ========================================================================
    // Model & Pattern Data
    // ========================================================================

    var SMART_DEFAULTS = {
        reasoner: 'grok-4.20-multi-agent-0309',
        coder: 'claude-opus-4-6',
        critic: 'gpt-5.4-2026-03-05',
        synthesizer: 'claude-sonnet-4-6',
    };

    var MODEL_GROUPS = [
        { label: 'X.AI', models: [
            { id: 'grok-4.20-multi-agent-0309', name: 'Grok Multi-Agent' },
            { id: 'grok-4.20-0309-reasoning', name: 'Grok Reasoning' },
            { id: 'grok-4.20-0309-non-reasoning', name: 'Grok Standard' },
        ]},
        { label: 'Anthropic', models: [
            { id: 'claude-opus-4-6', name: 'Claude Opus' },
            { id: 'claude-sonnet-4-6', name: 'Claude Sonnet' },
            { id: 'claude-haiku-4-5', name: 'Claude Haiku' },
        ]},
        { label: 'OpenAI', models: [
            { id: 'gpt-5.4-2026-03-05', name: 'GPT-5.4' },
            { id: 'gpt-5.4-mini-2026-03-17', name: 'GPT-5.4 Mini' },
        ]},
        { label: 'Google', models: [
            { id: 'gemini-3.1-pro-preview', name: 'Gemini Pro' },
            { id: 'gemini-3-flash-preview', name: 'Gemini Flash' },
        ]},
    ];

    var PATTERNS = {
        sequential: { label: 'Sequential', desc: 'Plan \u2192 Code \u2192 Review \u2192 Synthesize', steps: ['produce','code','review','synthesize'] },
        debate: { label: 'Debate', desc: 'Two perspectives \u2192 Reconcile', steps: ['debate_a','debate_b','reconcile'] },
        peer_review: { label: 'Peer Review', desc: 'Produce \u2192 Review \u2192 Rebut \u2192 React \u2192 Synth', steps: ['produce','review','rebuttal','react','synthesize'] },
    };

    var AGENT_COLORS = { reasoner: '#4a9eff', coder: '#3ddc84', critic: '#ff9f43', synthesizer: '#a855f7' };
    var ROLE_LABELS = { reasoner: 'Reasoner', coder: 'Coder', critic: 'Critic', synthesizer: 'Synthesizer' };

    // ========================================================================
    // Markdown / HTML Utilities (copied from chat.js)
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

    // ========================================================================
    // Safe DOM helpers — inject parsed markdown via DOM APIs, never raw HTML
    // ========================================================================

    /**
     * Parse the rendered markdown string and append its nodes to the given
     * container. Uses a temporary <span> as an inert fragment so the final
     * tree is built purely through appendChild.
     */
    function injectMarkdown(container, markdownText) {
        var rendered = renderMarkdown(markdownText);
        container.textContent = '';
        var wrapper = document.createElement('span');
        wrapper.innerHTML = rendered; // safe: renderMarkdown escapes untrusted parts
        while (wrapper.firstChild) {
            container.appendChild(wrapper.firstChild);
        }
    }

    // ========================================================================
    // Render Orchestrator
    // ========================================================================

    function render() {
        if (mode === 'config') {
            renderConfig();
        } else if (mode === 'live') {
            renderLive();
        } else if (mode === 'complete') {
            renderComplete();
        }
    }

    // ========================================================================
    // Config Mode
    // ========================================================================

    function renderConfig(prefill) {
        var app = document.getElementById('app');
        app.textContent = '';

        var questionVal = (prefill && prefill.question) ? prefill.question : '';
        var patternVal = (prefill && prefill.pattern) ? prefill.pattern : currentPattern;
        var budgetVal = (prefill && prefill.budget) ? prefill.budget : 2.0;

        // Root wrapper
        var root = document.createElement('div');
        root.className = 'config-mode active';

        // Title
        var title = document.createElement('span');
        title.className = 'section-label';
        title.textContent = '\uD83E\uDDE0 Deep Discussion';
        root.appendChild(title);

        // Question textarea
        var question = document.createElement('textarea');
        question.id = 'dd-question';
        question.className = 'question-input';
        question.placeholder = 'What do you want to analyze?';
        question.setAttribute('autofocus', '');
        question.value = questionVal;
        root.appendChild(question);

        // Pattern label
        var patLabel = document.createElement('span');
        patLabel.className = 'section-label';
        patLabel.textContent = 'Pattern';
        root.appendChild(patLabel);

        // Pattern select
        var patSelect = document.createElement('select');
        patSelect.id = 'dd-pattern';
        patSelect.className = 'pattern-select';
        var patternKeys = Object.keys(PATTERNS);
        for (var p = 0; p < patternKeys.length; p++) {
            var pk = patternKeys[p];
            var opt = document.createElement('option');
            opt.value = pk;
            opt.textContent = PATTERNS[pk].label + ' \u2014 ' + PATTERNS[pk].desc;
            if (pk === patternVal) { opt.selected = true; }
            patSelect.appendChild(opt);
        }
        root.appendChild(patSelect);

        // Start button
        var startBtn = document.createElement('button');
        startBtn.className = 'start-btn';
        startBtn.textContent = '\u25B6 Start Deep Discussion';
        startBtn.addEventListener('click', function() { window.handleStart(); });
        root.appendChild(startBtn);

        // Advanced toggle
        var advToggle = document.createElement('div');
        advToggle.className = advancedExpanded ? 'advanced-toggle open' : 'advanced-toggle';
        advToggle.addEventListener('click', function() { window.handleToggleAdvanced(); });
        var advIcon = document.createElement('span');
        advIcon.className = 'advanced-toggle-icon';
        advIcon.textContent = advancedExpanded ? '\u25BC' : '\u25B6';
        advToggle.appendChild(advIcon);
        var advText = document.createElement('span');
        advText.textContent = 'Advanced Configuration';
        advToggle.appendChild(advText);
        root.appendChild(advToggle);

        // Advanced section
        var advSection = document.createElement('div');
        advSection.className = advancedExpanded ? 'advanced-section' : 'advanced-section hidden';

        var roles = ['reasoner', 'coder', 'critic', 'synthesizer'];
        for (var r = 0; r < roles.length; r++) {
            var role = roles[r];
            var defaultModel = (prefill && prefill.modelAssignments && prefill.modelAssignments[role])
                ? prefill.modelAssignments[role]
                : SMART_DEFAULTS[role];

            var row = document.createElement('div');
            row.className = 'agent-config-row';

            var lbl = document.createElement('label');
            lbl.textContent = ROLE_LABELS[role];
            row.appendChild(lbl);

            var sel = document.createElement('select');
            sel.id = 'dd-model-' + role;
            for (var g = 0; g < MODEL_GROUPS.length; g++) {
                var group = MODEL_GROUPS[g];
                var optgroup = document.createElement('optgroup');
                optgroup.label = group.label;
                for (var m = 0; m < group.models.length; m++) {
                    var mOpt = document.createElement('option');
                    mOpt.value = group.models[m].id;
                    mOpt.textContent = group.models[m].name;
                    if (group.models[m].id === defaultModel) { mOpt.selected = true; }
                    optgroup.appendChild(mOpt);
                }
                sel.appendChild(optgroup);
            }
            row.appendChild(sel);
            advSection.appendChild(row);
        }

        // Budget row
        var budgetRow = document.createElement('div');
        budgetRow.className = 'budget-row';

        var budgetLabel = document.createElement('label');
        budgetLabel.textContent = 'Budget';
        budgetRow.appendChild(budgetLabel);

        var budgetInput = document.createElement('input');
        budgetInput.type = 'range';
        budgetInput.id = 'dd-budget';
        budgetInput.min = '0.5';
        budgetInput.max = '5';
        budgetInput.step = '0.25';
        budgetInput.value = String(budgetVal);
        budgetRow.appendChild(budgetInput);

        var budgetDisplay = document.createElement('span');
        budgetDisplay.className = 'budget-value';
        budgetDisplay.id = 'dd-budget-value';
        budgetDisplay.textContent = '$' + Number(budgetVal).toFixed(2);
        budgetRow.appendChild(budgetDisplay);

        budgetInput.addEventListener('input', function() {
            budgetDisplay.textContent = '$' + parseFloat(budgetInput.value).toFixed(2);
        });

        advSection.appendChild(budgetRow);
        root.appendChild(advSection);

        // History section
        if (historySessions.length > 0) {
            var histSection = document.createElement('div');
            histSection.className = 'history-section';

            var histTitle = document.createElement('div');
            histTitle.className = 'history-section-title';
            histTitle.textContent = 'Recent Sessions';
            histSection.appendChild(histTitle);

            for (var h = 0; h < historySessions.length; h++) {
                (function(session) {
                    var item = document.createElement('div');
                    item.className = 'history-item';
                    item.addEventListener('click', function() {
                        window.handleLoadSession(session.id);
                    });

                    var qDiv = document.createElement('div');
                    qDiv.className = 'history-item-question';
                    qDiv.textContent = session.question || 'Untitled';
                    item.appendChild(qDiv);

                    var metaDiv = document.createElement('div');
                    metaDiv.className = 'history-item-meta';
                    var metaParts = [];
                    if (session.pattern) { metaParts.push(session.pattern); }
                    if (session.cost) { metaParts.push('$' + session.cost.toFixed(4)); }
                    if (session.date) { metaParts.push(session.date); }
                    metaDiv.textContent = metaParts.join(' \u00b7 ');
                    item.appendChild(metaDiv);

                    histSection.appendChild(item);
                })(historySessions[h]);
            }

            root.appendChild(histSection);
        }

        app.appendChild(root);
    }

    // ========================================================================
    // Live Mode
    // ========================================================================

    function renderLive() {
        var app = document.getElementById('app');
        app.textContent = '';

        var pattern = PATTERNS[currentPattern] || PATTERNS.peer_review;

        var root = document.createElement('div');
        root.className = 'live-mode active';

        // Header
        var header = document.createElement('div');
        header.className = 'live-header';

        var patName = document.createElement('span');
        patName.className = 'live-pattern-name';
        patName.textContent = pattern.label;
        header.appendChild(patName);

        var status = document.createElement('span');
        status.className = 'live-status';
        status.textContent = '\u00b7 Running';
        header.appendChild(status);

        var costCounter = document.createElement('span');
        costCounter.className = 'live-cost-counter';
        costCounter.textContent = '$' + totalCost.toFixed(4);
        header.appendChild(costCounter);

        root.appendChild(header);

        // Cancel link
        var cancel = document.createElement('span');
        cancel.className = 'cancel-link';
        cancel.textContent = 'Cancel';
        cancel.addEventListener('click', function() { window.handleCancel(); });
        root.appendChild(cancel);

        // Progress bar
        appendProgressBar(root, pattern);

        // Agent rows
        appendAgentRows(root);

        app.appendChild(root);

        // Hydrate expanded agent content
        hydrateAgentContent();
    }

    // ========================================================================
    // Complete Mode
    // ========================================================================

    function renderComplete() {
        var app = document.getElementById('app');
        app.textContent = '';

        var pattern = PATTERNS[currentPattern] || PATTERNS.peer_review;

        var root = document.createElement('div');
        root.className = 'complete-mode active';

        // Header
        var header = document.createElement('div');
        header.className = 'complete-header';

        var titleSpan = document.createElement('span');
        titleSpan.className = 'complete-title';
        titleSpan.textContent = pattern.label;
        header.appendChild(titleSpan);

        var meta = document.createElement('span');
        meta.className = 'complete-meta';
        meta.textContent = '\u00b7 Done \u00b7 $' + totalCost.toFixed(4);
        header.appendChild(meta);

        root.appendChild(header);

        // Full green progress bar (all steps done)
        var progressBar = document.createElement('div');
        progressBar.className = 'progress-bar';
        var labelsRow = document.createElement('div');
        labelsRow.className = 'step-labels';
        for (var s = 0; s < pattern.steps.length; s++) {
            var stepDiv = document.createElement('div');
            stepDiv.className = 'progress-step done';
            progressBar.appendChild(stepDiv);

            var labelDiv = document.createElement('div');
            labelDiv.className = 'step-label';
            labelDiv.textContent = pattern.steps[s];
            labelsRow.appendChild(labelDiv);
        }
        root.appendChild(progressBar);
        root.appendChild(labelsRow);

        // Final answer
        var finalAnswer = document.createElement('div');
        finalAnswer.className = 'final-answer';
        finalAnswer.id = 'dd-final-answer';
        if (sessionResult) {
            injectMarkdown(finalAnswer, sessionResult);
        }
        root.appendChild(finalAnswer);

        // Agent rows
        appendAgentRows(root);

        // Run Again button
        var runAgainBtn = document.createElement('button');
        runAgainBtn.className = 'run-again-btn';
        runAgainBtn.textContent = '\u21BB Run Again';
        runAgainBtn.addEventListener('click', function() { window.handleRunAgain(); });
        root.appendChild(runAgainBtn);

        app.appendChild(root);

        // Hydrate expanded agent content
        hydrateAgentContent();
    }

    // ========================================================================
    // Progress Bar (DOM builder)
    // ========================================================================

    function appendProgressBar(parent, pattern) {
        var steps = pattern.steps;

        var bar = document.createElement('div');
        bar.className = 'progress-bar';
        var labels = document.createElement('div');
        labels.className = 'step-labels';

        for (var i = 0; i < steps.length; i++) {
            var step = steps[i];
            var stepDiv = document.createElement('div');
            if (completedSteps.indexOf(step) !== -1) {
                stepDiv.className = 'progress-step done';
            } else if (step === currentStep) {
                stepDiv.className = 'progress-step active';
            } else {
                stepDiv.className = 'progress-step pending';
            }
            bar.appendChild(stepDiv);

            var labelDiv = document.createElement('div');
            labelDiv.className = 'step-label';
            labelDiv.textContent = step;
            labels.appendChild(labelDiv);
        }

        parent.appendChild(bar);
        parent.appendChild(labels);
    }

    // ========================================================================
    // Agent Rows (DOM builder)
    // ========================================================================

    function appendAgentRows(parent) {
        for (var i = 0; i < agents.length; i++) {
            appendAgentRow(parent, agents[i], i);
        }
    }

    function appendAgentRow(parent, agent, index) {
        var color = AGENT_COLORS[agent.role] || '#888';
        var label = ROLE_LABELS[agent.role] || agent.role;

        // Container wraps the clickable row and expandable content
        var container = document.createElement('div');
        container.style.marginBottom = '6px';

        // Row
        var row = document.createElement('div');
        row.className = agent.expanded ? 'agent-row expanded' : 'agent-row';
        (function(idx) {
            row.addEventListener('click', function() { window.handleToggleAgent(idx); });
        })(index);

        // Color dot
        var dot = document.createElement('span');
        dot.className = 'agent-dot';
        dot.style.backgroundColor = color;
        row.appendChild(dot);

        // Name
        var nameSpan = document.createElement('span');
        nameSpan.className = 'agent-name';
        nameSpan.style.color = color;
        nameSpan.textContent = label;
        row.appendChild(nameSpan);

        // Model hint
        if (agent.model) {
            var modelSpan = document.createElement('span');
            modelSpan.style.fontSize = '10px';
            modelSpan.style.opacity = '0.5';
            modelSpan.style.flexShrink = '0';
            modelSpan.textContent = agent.model;
            row.appendChild(modelSpan);
        }

        // Status
        var statusSpan = document.createElement('span');
        statusSpan.className = 'agent-status';
        if (agent.status === 'thinking') {
            statusSpan.className += ' thinking';
            statusSpan.textContent = 'thinking\u2026';
        } else if (agent.status === 'done') {
            statusSpan.className += ' done';
            statusSpan.textContent = 'done';
        } else {
            statusSpan.className += ' pending';
            statusSpan.textContent = agent.status || 'pending';
        }
        row.appendChild(statusSpan);

        // Cost
        if (agent.cost && agent.cost > 0) {
            var costSpan = document.createElement('span');
            costSpan.className = 'agent-cost';
            costSpan.textContent = '$' + agent.cost.toFixed(4);
            row.appendChild(costSpan);
        }

        container.appendChild(row);

        // Expandable content area
        var contentDiv = document.createElement('div');
        contentDiv.className = 'agent-content';
        contentDiv.id = 'dd-agent-content-' + index;
        if (agent.expanded) {
            contentDiv.style.display = 'block';
        }
        container.appendChild(contentDiv);

        parent.appendChild(container);
    }

    /**
     * After a full render, inject markdown content into agent content divs.
     * This avoids building HTML strings for markdown rendering.
     */
    function hydrateAgentContent() {
        for (var i = 0; i < agents.length; i++) {
            if (agents[i].content && agents[i].expanded) {
                var el = document.getElementById('dd-agent-content-' + i);
                if (el) {
                    injectMarkdown(el, agents[i].content);
                }
            }
        }
    }

    // ========================================================================
    // Event Handlers (exposed globally for inline onclick)
    // ========================================================================

    window.handleStart = function() {
        var questionEl = document.getElementById('dd-question');
        var patternEl = document.getElementById('dd-pattern');
        if (!questionEl) return;

        var question = questionEl.value.trim();
        if (!question) {
            questionEl.focus();
            return;
        }

        var pattern = patternEl ? patternEl.value : 'peer_review';
        currentPattern = pattern;

        // Disable start button
        var startBtn = document.querySelector('.start-btn');
        if (startBtn) {
            startBtn.disabled = true;
            startBtn.textContent = 'Starting\u2026';
        }

        // Collect model assignments from advanced config
        var modelAssignments = {};
        var roles = ['reasoner', 'coder', 'critic', 'synthesizer'];
        for (var i = 0; i < roles.length; i++) {
            var selectEl = document.getElementById('dd-model-' + roles[i]);
            if (selectEl) {
                modelAssignments[roles[i]] = selectEl.value;
            } else {
                modelAssignments[roles[i]] = SMART_DEFAULTS[roles[i]];
            }
        }

        // Collect budget
        var budgetEl = document.getElementById('dd-budget');
        var budget = budgetEl ? parseFloat(budgetEl.value) : 2.0;

        vscode.postMessage({
            type: 'startSession',
            question: question,
            pattern: pattern,
            modelAssignments: modelAssignments,
            budget: budget
        });
    };

    window.handleCancel = function() {
        vscode.postMessage({ type: 'cancelSession' });
    };

    window.handleRunAgain = function() {
        vscode.postMessage({ type: 'runAgain' });
    };

    window.handleToggleAgent = function(index) {
        if (index >= 0 && index < agents.length) {
            agents[index].expanded = !agents[index].expanded;
            render();
        }
    };

    window.handleToggleAdvanced = function() {
        advancedExpanded = !advancedExpanded;
        // Preserve current form values before re-render
        var questionEl = document.getElementById('dd-question');
        var questionVal = questionEl ? questionEl.value : '';
        var patternEl = document.getElementById('dd-pattern');
        var patternVal = patternEl ? patternEl.value : currentPattern;
        var budgetEl = document.getElementById('dd-budget');
        var budgetVal = budgetEl ? budgetEl.value : 2.0;

        // Collect current model selections if they exist
        var modelAssignments = {};
        var roles = ['reasoner', 'coder', 'critic', 'synthesizer'];
        for (var i = 0; i < roles.length; i++) {
            var selectEl = document.getElementById('dd-model-' + roles[i]);
            if (selectEl) {
                modelAssignments[roles[i]] = selectEl.value;
            }
        }

        renderConfig({
            question: questionVal,
            pattern: patternVal,
            budget: budgetVal,
            modelAssignments: Object.keys(modelAssignments).length > 0 ? modelAssignments : null
        });
    };

    window.handleLoadSession = function(sessionId) {
        vscode.postMessage({ type: 'loadSession', sessionId: sessionId });
    };

    // ========================================================================
    // Message Handler (from extension)
    // ========================================================================

    window.addEventListener('message', function(event) {
        var msg = event.data;

        switch (msg.type) {
            case 'showConfig':
                mode = 'config';
                agents = [];
                completedSteps = [];
                currentStep = '';
                totalCost = 0;
                sessionResult = '';
                currentPattern = 'peer_review';
                renderConfig(msg.prefill || null);
                break;

            case 'showLive':
                mode = 'live';
                agents = [];
                completedSteps = [];
                currentStep = '';
                totalCost = 0;
                sessionResult = '';
                if (msg.pattern) {
                    currentPattern = msg.pattern;
                }
                render();
                break;

            case 'sessionStart':
                if (msg.pattern) {
                    currentPattern = msg.pattern;
                }
                break;

            case 'stepStart':
                currentStep = msg.step;
                render();
                break;

            case 'agentStart':
                agents.push({
                    role: msg.role,
                    model: msg.model || '',
                    status: 'thinking',
                    content: '',
                    cost: 0,
                    tokens: 0,
                    expanded: false
                });
                render();
                break;

            case 'agentChunk':
                // Find the thinking agent matching this role (search from end)
                for (var i = agents.length - 1; i >= 0; i--) {
                    if (agents[i].role === msg.role && agents[i].status === 'thinking') {
                        agents[i].content += (msg.content || '');
                        break;
                    }
                }
                render();
                break;

            case 'agentDone':
                // Find the thinking agent matching this role (search from end)
                for (var j = agents.length - 1; j >= 0; j--) {
                    if (agents[j].role === msg.role && agents[j].status === 'thinking') {
                        agents[j].status = 'done';
                        agents[j].cost = msg.cost || 0;
                        agents[j].tokens = msg.tokens || 0;
                        totalCost += agents[j].cost;
                        break;
                    }
                }
                render();
                break;

            case 'stepComplete':
                if (msg.step && completedSteps.indexOf(msg.step) === -1) {
                    completedSteps.push(msg.step);
                }
                currentStep = '';
                render();
                break;

            case 'sessionComplete':
                mode = 'complete';
                sessionResult = msg.result || '';
                if (msg.totalCost !== undefined) {
                    totalCost = msg.totalCost;
                }
                render();
                break;

            case 'sessionError':
                showError(msg.message || 'An unknown error occurred.');
                break;

            case 'historyData':
                historySessions = msg.sessions || [];
                if (mode === 'config') {
                    renderConfig();
                }
                break;

            case 'sessionData':
                // Reconstruct complete view from saved session data
                mode = 'complete';
                currentPattern = msg.pattern || 'peer_review';
                sessionResult = msg.result || '';
                totalCost = msg.totalCost || 0;
                agents = (msg.agents || []).map(function(a) {
                    return {
                        role: a.role || '',
                        model: a.model || '',
                        status: 'done',
                        content: a.content || '',
                        cost: a.cost || 0,
                        tokens: a.tokens || 0,
                        expanded: false
                    };
                });
                completedSteps = (PATTERNS[currentPattern] || PATTERNS.peer_review).steps.slice();
                currentStep = '';
                render();
                break;
        }
    });

    // ========================================================================
    // Error Display
    // ========================================================================

    function showError(message) {
        var app = document.getElementById('app');
        // Remove any existing error before adding a new one
        var existing = app.querySelector('.error-message');
        if (existing) {
            existing.remove();
        }
        var errDiv = document.createElement('div');
        errDiv.className = 'error-message';
        errDiv.textContent = 'Error: ' + message;
        app.appendChild(errDiv);
    }

    // ========================================================================
    // Initial Setup
    // ========================================================================

    vscode.postMessage({ type: 'loadHistory' });
    render();
})();
