IFRAME_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Chat With Us</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Google Fonts for Arabic and Latin support -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;700&family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <!-- Marked.js for Markdown parsing -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
    <!-- Highlight.js for code syntax highlighting -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-light.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <style>
        :root {
            /* Light mode variables */
            --bg-color: #f0f2f5;
            --bg-gradient: linear-gradient(135deg, #f8f9fa, #e0e5ec);
            --chat-bg: white;
            --input-bg: #f8f9fa;
            --border-color: #e9ecef;
            --user-bg: #4e7df9;
            --user-text: white;
            --bot-bg: #f0f2f5;
            --bot-text: #333;
            --text-color: #333;
            --shadow: rgba(0,0,0,0.1);
            --input-border: #ced4da;
            --font-arabic: 'IBM Plex Sans Arabic', sans-serif;
            --font-latin: 'Inter', sans-serif;
            --accent-color: #4e7df9;
            --info-color: #17a2b8;
            --placeholder-color: #adb5bd;
            --code-bg: #f5f7f9;
            --code-border: #eaecef;
            --blockquote-bg: #f8f9fa;
            --blockquote-border: #dee2e6;
            --progress-bg: #e0e0e0;
            --progress-fill: linear-gradient(90deg, #ff4444, #ff8787);
        }
        [data-theme="dark"] {
            /* Dark mode variables */
            --bg-color: #1a1a1a;
            --bg-gradient: linear-gradient(135deg, #1a1a1a, #2d2d2d);
            --chat-bg: #2d2d2d;
            --input-bg: #3a3a3a;
            --border-color: #444;
            --user-bg: #4e7df9;
            --user-text: white;
            --bot-bg: #444;
            --bot-text: #ddd;
            --text-color: #ddd;
            --shadow: rgba(0,0,0,0.3);
            --input-border: #555;
            --placeholder-color: #888;
            --code-bg: #3a3a3a;
            --code-border: #555;
            --blockquote-bg: #3a3a3a;
            --blockquote-border: #555;
            --progress-bg: #555;
            --progress-fill: linear-gradient(90deg, #ff6666, #ff9999);
        }
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: var(--font-latin);
            background: var(--bg-gradient);
            display: flex;
            flex-direction: column;
            color: var(--text-color);
            overflow: hidden;
            transition: background 0.3s ease;
        }
        .rtl-layout {
            direction: rtl;
            font-family: var(--font-arabic);
        }
        #chat-container {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            background: var(--chat-bg);
            position: relative;
            font-size: 0.95em;
            box-sizing: border-box;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        #header {
            padding: 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: var(--accent-color);
            color: white;
            box-shadow: 0 2px 5px var(--shadow);
            z-index: 10;
        }
        .header-title {
            font-weight: 600;
            font-size: 1.2em;
            display: flex;
            align-items: center;
        }
        .header-title i {
            margin-right: 8px;
            font-size: 1.2em;
        }
        .rtl-layout .header-title i {
            margin-right: 0;
            margin-left: 8px;
        }
        .header-controls {
            display: flex;
            gap: 15px;
        }
        #theme-toggle, #lang-toggle {
            background: transparent;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            padding: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.3s ease;
        }
        #theme-toggle:hover, #lang-toggle:hover {
            transform: rotate(30deg);
        }
        #messages {
            flex: 1;
            overflow-y: auto;
            padding: 15px;
            line-height: 1.5;
            box-sizing: border-box;
            scroll-behavior: smooth;
        }
        #messages::-webkit-scrollbar {
            width: 8px;
        }
        #messages::-webkit-scrollbar-track {
            background: transparent;
        }
        #messages::-webkit-scrollbar-thumb {
            background-color: var(--border-color);
            border-radius: 20px;
        }
        #input-form {
            display: flex;
            padding: 15px;
            background: var(--input-bg);
            border-top: 1px solid var(--border-color);
            box-sizing: border-box;
            position: relative;
            transition: background-color 0.3s ease;
        }
        #query-wrapper {
            flex: 1;
            position: relative;
            display: flex;
            align-items: center;
        }
        #query {
            flex: 1;
            padding: 12px 15px;
            border: 1px solid var(--input-border);
            border-radius: 20px;
            outline: none;
            background: var(--chat-bg);
            color: var(--text-color);
            transition: border-color 0.3s ease, box-shadow 0.3s ease, opacity 0.3s ease;
            direction: ltr;
            text-align: left;
            font-family: var(--font-latin);
            font-size: 1em;
            box-sizing: border-box;
            height: 46px;
            width: 100%;
        }
        #query.hidden {
            display: none;
        }
        #query::placeholder {
            color: var(--placeholder-color);
            transition: opacity 0.3s ease;
        }
        #query:focus::placeholder {
            opacity: 0.7;
        }
        #query.rtl {
            direction: rtl;
            text-align: right;
            font-family: var(--font-arabic);
        }
        #query:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(78, 125, 249, 0.2);
        }
        #recording-progress {
            display: none;
            flex: 1;
            height: 46px;
            background: var(--progress-bg);
            border-radius: 20px;
            overflow: hidden;
            position: relative;
        }
        #recording-progress.recording {
            display: block;
        }
        #recording-progress::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            width: 0;
            background: var(--progress-fill);
            animation: progress 60s linear forwards;
            transition: width 0.1s ease;
        }
        @keyframes progress {
            0% { width: 0; }
            100% { width: 100%; }
        }
        button[type="submit"], #record-btn {
            width: 46px;
            height: 46px;
            margin-left: 10px;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 5px var(--shadow);
        }
        .rtl-layout button[type="submit"], .rtl-layout #record-btn {
            margin-left: 0;
            margin-right: 10px;
        }
        button[type="submit"]:hover, #record-btn:hover {
            background: #3a6ae0;
            transform: scale(1.05);
        }
        button[type="submit"]:active, #record-btn:active {
            transform: scale(0.95);
        }
        #record-btn.recording {
            background: #ff4444;
        }
        .message {
            margin: 10px 0;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 85%;
            width: fit-content;
            min-width: 80px;
            word-wrap: break-word;
            display: block;
            font-family: var(--font-latin);
            box-shadow: 0 1px 2px var(--shadow);
            position: relative;
            opacity: 0;
            transform: translateY(20px);
            animation: fadeIn 0.3s ease-out forwards;
        }
        .message.rtl {
            direction: rtl;
            text-align: right;
            font-family: var(--font-arabic);
        }
        .user-message {
            background: var(--user-bg);
            color: var(--user-text);
            margin-left: auto;
            text-align: left;
            border-bottom-right-radius: 5px;
        }
        .rtl-layout .user-message {
            margin-left: 0;
            margin-right: auto;
            border-bottom-right-radius: 18px;
            border-bottom-left-radius: 5px;
        }
        .bot-message {
            background: var(--bot-bg);
            color: var(--bot-text);
            margin-right: auto;
            text-align: left;
            border-bottom-left-radius: 5px;
        }
        .rtl-layout .bot-message {
            margin-right: 0;
            margin-left: auto;
            border-bottom-left-radius: 18px;
            border-bottom-right-radius: 5px;
        }
        .message-content {
            width: 100%;
        }
        .bot-message .message-content strong { font-weight: 700; }
        .bot-message .message-content em { font-style: italic; }
        .bot-message .message-content h1,
        .bot-message .message-content h2,
        .bot-message .message-content h3,
        .bot-message .message-content h4,
        .bot-message .message-content h5,
        .bot-message .message-content h6 {
            margin-top: 10px;
            margin-bottom: 5px;
            font-weight: 600;
            line-height: 1.3;
        }
        .bot-message .message-content h1 {
            font-size: 1.4em;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 5px;
        }
        .bot-message .message-content h2 { font-size: 1.3em; }
        .bot-message .message-content h3 { font-size: 1.2em; }
        .bot-message .message-content h4 { font-size: 1.1em; }
        .bot-message .message-content p { margin: 8px 0; }
        .bot-message .message-content ul,
        .bot-message .message-content ol {
            margin: 8px 0;
            padding-left: 20px;
        }
        .rtl-layout .bot-message .message-content ul,
        .rtl-layout .bot-message .message-content ol {
            padding-left: 0;
            padding-right: 20px;
        }
        .bot-message .message-content li { margin: 4px 0; }
        .bot-message .message-content a {
            color: var(--accent-color);
            text-decoration: none;
        }
        .bot-message .message-content a:hover { text-decoration: underline; }
        .bot-message .message-content code {
            background-color: var(--code-bg);
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.9em;
        }
        .bot-message .message-content pre {
            background-color: var(--code-bg);
            border: 1px solid var(--code-border);
            border-radius: 5px;
            padding: 10px;
            overflow-x: auto;
            margin: 8px 0;
        }
        .bot-message .message-content pre code {
            background-color: transparent;
            padding: 0;
            border-radius: 0;
            display: block;
            font-size: 0.85em;
            line-height: 1.5;
        }
        .bot-message .message-content blockquote {
            border-left: 4px solid var(--accent-color);
            margin: 8px 0;
            padding: 8px 12px;
            background-color: var(--blockquote-bg);
            border-radius: 0 5px 5px 0;
        }
        .rtl-layout .bot-message .message-content blockquote {
            border-left: none;
            border-right: 4px solid var(--accent-color);
            border-radius: 5px 0 0 5px;
        }
        .bot-message .message-content blockquote p { margin: 5px 0; }
        .bot-message .message-content table {
            border-collapse: collapse;
            margin: 10px 0;
            width: 100%;
        }
        .bot-message .message-content th,
        .bot-message .message-content td {
            border: 1px solid var(--border-color);
            padding: 6px 10px;
        }
        .bot-message .message-content th {
            background-color: var(--bot-bg);
            font-weight: 600;
        }
        .typing {
            color: #888;
            font-style: italic;
            background: var(--bot-bg);
        }
        .typing .dot {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background-color: #888;
            margin-right: 3px;
            animation: dotPulse 1.5s infinite;
        }
        .typing .dot:nth-child(2) { animation-delay: 0.2s; }
        .typing .dot:nth-child(3) { animation-delay: 0.4s; }
        .message-time {
            font-size: 0.7em;
            opacity: 0.7;
            margin-top: 5px;
            display: block;
        }
        .clear-button {
            padding: 6px 12px;
            background: transparent;
            color: white;
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 15px;
            cursor: pointer;
            font-size: 0.8em;
            transition: all 0.3s ease;
        }
        .clear-button:hover {
            background: rgba(255,255,255,0.1);
        }
        .message-actions {
            position: absolute;
            right: 10px;
            top: -20px;
            display: none;
            background: var(--chat-bg);
            border-radius: 15px;
            padding: 3px 8px;
            box-shadow: 0 1px 3px var(--shadow);
            font-size: 0.8em;
            gap: 8px;
        }
        .rtl-layout .message-actions {
            right: auto;
            left: 10px;
        }
        .message:hover .message-actions { display: flex; }
        .message-action {
            cursor: pointer;
            color: var(--placeholder-color);
            transition: color 0.2s;
        }
        .message-action:hover { color: var(--accent-color); }
        @keyframes dotPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.5); opacity: 0.5; }
        }
        @keyframes fadeIn {
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .welcome-message {
            text-align: center;
            padding: 20px;
            color: var(--placeholder-color);
            animation: slideUp 0.5s ease-out forwards;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 20px auto;
        }
        .welcome-icon {
            font-size: 3em;
            margin-bottom: 15px;
            color: var(--accent-color);
        }
        .welcome-title {
            font-size: 1.5em;
            margin-bottom: 10px;
            font-weight: 600;
            color: var(--text-color);
        }
        .welcome-subtitle {
            font-size: 1em;
            margin-bottom: 20px;
            max-width: 80%;
        }
        .suggestions {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-top: 15px;
            max-width: 90%;
        }
        .suggestion {
            padding: 8px 15px;
            background: var(--bot-bg);
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.9em;
            border: 1px solid var(--border-color);
        }
        .suggestion:hover {
            background: var(--accent-color);
            color: white;
            transform: translateY(-2px);
        }
        [data-theme="dark"] .hljs { background: var(--code-bg); color: #e6e6e6; }
        [data-theme="dark"] .hljs-keyword,
        [data-theme="dark"] .hljs-selector-tag,
        [data-theme="dark"] .hljs-title,
        [data-theme="dark"] .hljs-section,
        [data-theme="dark"] .hljs-doctag,
        [data-theme="dark"] .hljs-name,
        [data-theme="dark"] .hljs-strong { color: #569cd6; }
        [data-theme="dark"] .hljs-string,
        [data-theme="dark"] .hljs-attr,
        [data-theme="dark"] .hljs-symbol,
        [data-theme="dark"] .hljs-bullet,
        [data-theme="dark"] .hljs-built_in,
        [data-theme="dark"] .hljs-builtin-name,
        [data-theme="dark"] .hljs-comment,
        [data-theme="dark"] .hljs-quote,
        [data-theme="dark"] .hljs-meta,
        [data-theme="dark"] .hljs-deletion { color: #ce9178; }
        [data-theme="dark"] .hljs-number,
        [data-theme="dark"] .hljs-regexp,
        [data-theme="dark"] .hljs-literal,
        [data-theme="dark"] .hljs-link,
        [data-theme="dark"] .hljs-meta,
        [data-theme="dark"] .hljs-addition { color: #b5cea8; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="header">
            <div class="header-title">
                <i class="fas fa-comments"></i>
                <span id="header-title-text">Chat Assistant</span>
            </div>
            <div class="header-controls">
                <button class="clear-button" id="clear-chat">
                    <i class="fas fa-trash-alt"></i> <span id="clear-text">Clear</span>
                </button>
                <button id="lang-toggle">🇬🇧</button>
                <button id="theme-toggle">☀️</button>
            </div>
        </div>
        <div id="messages">
            <!-- Welcome message will be added here -->
        </div>
        <form id="input-form">
            <div id="query-wrapper">
                <input type="text" id="query" placeholder="Type your message..." data-arabic-placeholder="اكتب رسالتك هنا...">
                <div id="recording-progress"></div>
            </div>
            <button type="button" id="record-btn"><i class="fas fa-microphone"></i></button>
            <button type="submit"><i class="fas fa-paper-plane"></i></button>
        </form>
    </div>

    <script>
        const form = document.getElementById('input-form');
        const messages = document.getElementById('messages');
        const queryInput = document.getElementById('query');
        const themeToggle = document.getElementById('theme-toggle');
        const langToggle = document.getElementById('lang-toggle');
        const clearChatBtn = document.getElementById('clear-chat');
        const recordBtn = document.getElementById('record-btn');
        const recordingProgress = document.getElementById('recording-progress');
        const body = document.body;
        const container = document.getElementById('chat-container');
        const headerTitleText = document.getElementById('header-title-text');
        const clearText = document.getElementById('clear-text');

        const translations = {
            en: {
                headerTitle: "Chat Assistant",
                clearButton: "Clear",
                inputPlaceholder: "Type your message...",
                welcomeTitle: "Hello! How can I help you today?",
                welcomeSubtitle: "Ask me anything or try one of these examples:",
                suggestion1: "Tell me about your services",
                suggestion2: "About the company",
                suggestion3: "Contact information"
            },
            ar: {
                headerTitle: "المساعد الذكى",
                clearButton: "مسح",
                inputPlaceholder: "اكتب رسالتك هنا...",
                welcomeTitle: "مرحباً! كيف يمكنني مساعدتك اليوم؟",
                welcomeSubtitle: "اسألني أي شيء أو جرب أحد هذه الأمثلة:",
                suggestion1: "أخبرني عن خدماتكم",
                suggestion2: "عن الشركة",
                suggestion3: "معلومات الاتصال"
            }
        };

        marked.setOptions({
            gfm: true,
            breaks: true,
            pedantic: false,
            sanitize: false,
            smartLists: true,
            smartypants: false,
            highlight: function(code, lang) {
                if (lang && hljs.getLanguage(lang)) {
                    return hljs.highlight(code, { language: lang }).value;
                } else {
                    return hljs.highlightAuto(code).value;
                }
            }
        });

        let currentLang = localStorage.getItem('language') || 'en';
        applyLanguage(currentLang);

        window.onload = function() {
            if (messages.children.length === 0) {
                showWelcomeMessage();
            }
        };

        function showWelcomeMessage() {
            const langData = translations[currentLang];
            const welcomeDiv = document.createElement('div');
            welcomeDiv.className = 'welcome-message';
            welcomeDiv.innerHTML = `
                <div class="welcome-icon"><i class="fas fa-robot"></i></div>
                <div class="welcome-title">${langData.welcomeTitle}</div>
                <div class="welcome-subtitle">${langData.welcomeSubtitle}</div>
                <div class="suggestions">
                    <div class="suggestion">${langData.suggestion1}</div>
                    <div class="suggestion">${langData.suggestion2}</div>
                    <div class="suggestion">${langData.suggestion3}</div>
                </div>
            `;
            messages.appendChild(welcomeDiv);

            const suggestions = document.querySelectorAll('.suggestion');
            suggestions.forEach(suggestion => {
                suggestion.addEventListener('click', () => {
                    queryInput.value = suggestion.textContent;
                    form.dispatchEvent(new Event('submit'));
                });
            });
        }

        function isArabic(text) {
            const arabic = /[\u0600-\u06FF]/;
            return arabic.test(text);
        }

        function applyLanguage(lang) {
            currentLang = lang;
            localStorage.setItem('language', lang);
            const langData = translations[lang];
            headerTitleText.textContent = langData.headerTitle;
            clearText.textContent = langData.clearButton;
            queryInput.placeholder = langData.inputPlaceholder;

            if (lang === 'ar') {
                langToggle.textContent = 'en';
                document.body.classList.add('rtl-layout');
                container.dir = 'rtl';
            } else {
                langToggle.textContent = 'ع';
                document.body.classList.remove('rtl-layout');
                container.dir = 'ltr';
            }

            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) {
                messages.removeChild(welcomeMessage);
                showWelcomeMessage();
            }
        }

        const savedTheme = localStorage.getItem('theme');
        if (savedTheme === 'dark') {
            body.setAttribute('data-theme', 'dark');
            themeToggle.textContent = '🌙';
        } else {
            themeToggle.textContent = '☀️';
        }

        langToggle.onclick = () => {
            const newLang = currentLang === 'en' ? 'ar' : 'en';
            applyLanguage(newLang);
        };

        themeToggle.onclick = () => {
            if (body.getAttribute('data-theme') === 'dark') {
                body.removeAttribute('data-theme');
                localStorage.setItem('theme', 'light');
                themeToggle.textContent = '☀️';
            } else {
                body.setAttribute('data-theme', 'dark');
                localStorage.setItem('theme', 'dark');
                themeToggle.textContent = '🌙';
            }
        };

        clearChatBtn.onclick = () => {
            messages.innerHTML = '';
            showWelcomeMessage();
        };

        queryInput.addEventListener('input', () => {
            const text = queryInput.value;
            const isRtl = isArabic(text);
            if (isRtl) {
                queryInput.classList.add('rtl');
                queryInput.classList.add('arabic-text');
                queryInput.setAttribute('placeholder', translations.ar.inputPlaceholder);
            } else {
                queryInput.classList.remove('rtl');
                queryInput.classList.remove('arabic-text');
                queryInput.setAttribute('placeholder', translations.en.inputPlaceholder);
            }
        });

        async function sendQuery(query) {
            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) messages.removeChild(welcomeMessage);

            addMessage(query, 'user-message', isArabic(query));
            const typing = addMessage('<div class="dot"></div><div class="dot"></div><div class="dot"></div>', 'bot-message typing');

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, language: currentLang })
                });
                const data = await response.json();
                messages.removeChild(typing);

                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot-message');
                } else {
                    addMessage(data.response, 'bot-message', isArabic(data.response) || currentLang === 'ar');
                }
            } catch (error) {
                messages.removeChild(typing);
                addMessage('Error: ' + error.message, 'bot-message');
            }
        }

        form.onsubmit = async (e) => {
            e.preventDefault();
            const query = queryInput.value.trim();
            if (!query) return;
            queryInput.value = '';
            await sendQuery(query);
        };

        let mediaRecorder;
        let audioChunks = [];

        function startRecordingProgress() {
            queryInput.classList.add('hidden');
            recordingProgress.classList.add('recording');
            recordingProgress.style.animation = 'none'; // Reset animation
            void recordingProgress.offsetWidth; // Trigger reflow
            recordingProgress.style.animation = null; // Reapply animation
        }

        function stopRecordingProgress() {
            queryInput.classList.remove('hidden');
            recordingProgress.classList.remove('recording');
        }

        recordBtn.addEventListener('click', async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                recordBtn.classList.remove('recording');
                recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
                stopRecordingProgress();
            } else {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    audioChunks = [];
                    mediaRecorder.addEventListener('dataavailable', event => {
                        audioChunks.push(event.data);
                    });
                    mediaRecorder.addEventListener('stop', async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
                        const formData = new FormData();
                        formData.append('file', audioFile);

                        try {
                            const response = await fetch('/transcribe', {
                                method: 'POST',
                                body: formData
                            });
                            const data = await response.json();
                            if (data.transcription) {
                                await sendQuery(data.transcription);
                            } else {
                                addMessage('Error: ' + data.error, 'bot-message');
                            }
                        } catch (error) {
                            console.error('Transcription error:', error);
                            addMessage('Error transcribing audio', 'bot-message');
                        }

                        stream.getTracks().forEach(track => track.stop());
                    });
                    mediaRecorder.start();
                    recordBtn.classList.add('recording');
                    recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    startRecordingProgress();
                } catch (error) {
                    console.error('Error accessing microphone:', error);
                    addMessage('Error: Could not access microphone', 'bot-message');
                }
            }
        });

        function formatTime() {
            const now = new Date();
            return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }

        function addMessage(text, className, isRtl = false) {
            const div = document.createElement('div');
            div.className = `message ${className}${isRtl ? ' rtl arabic-text' : ''}`;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';

            if (className === 'bot-message' && !className.includes('typing')) {
                contentDiv.innerHTML = marked.parse(text);
                contentDiv.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
            } else {
                contentDiv.textContent = text;
            }

            if (!className.includes('typing')) {
                div.appendChild(contentDiv);
                const timeSpan = document.createElement('span');
                timeSpan.className = 'message-time';
                timeSpan.textContent = formatTime();
                div.appendChild(timeSpan);

                if (className === 'bot-message') {
                    const actionsDiv = document.createElement('div');
                    actionsDiv.className = 'message-actions';
                    actionsDiv.innerHTML = '<span class="message-action" title="Copy"><i class="fas fa-copy"></i></span>';
                    div.appendChild(actionsDiv);
                }
            } else {
                div.innerHTML = text;
            }

            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;

            if (!className.includes('typing')) {
                const copyBtn = div.querySelector('.message-action');
                if (copyBtn) {
                    copyBtn.addEventListener('click', () => {
                        navigator.clipboard.writeText(text).then(() => {
                            copyBtn.innerHTML = '<i class="fas fa-check"></i>';
                            setTimeout(() => copyBtn.innerHTML = '<i class="fas fa-copy"></i>', 2000);
                        });
                    });
                }
            }

            return div;
        }
    </script>
</body>
</html>
"""
