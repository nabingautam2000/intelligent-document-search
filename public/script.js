// script.js - Full Claude Chat Interface Logic

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const sidebarSearchInput = document.getElementById('sidebarSearchInput');
    const chatList = document.getElementById('chatList');
    const newChatFloatingBtn = document.getElementById('newChatFloatingBtn'); // The floating button
    const notificationBar = document.getElementById('notificationBar');
    const closeNotificationBtn = document.getElementById('closeNotificationBtn');
    const mainChatArea = document.getElementById('mainChatArea'); // Parent of welcome/conversation
    const welcomeMessageDiv = document.getElementById('welcomeMessage'); // "How can Claude help you today?"
    const conversationMessagesDiv = document.getElementById('conversationMessages'); // Actual message bubbles container
    const chatInput = document.getElementById('chatInput');
    const sendMessageBtn = document.getElementById('sendMessageBtn'); // Send button for live chat

    let allChatSessions = []; // Stores all fetched sessions for filtering in sidebar
    let currentActiveConversationMessages = []; // Messages for the actively viewed chat (excluding system message)
    let activeChatId = null; // ID of the currently active chat session

    // --- Utility Functions ---

    // Function to format time for chat items
    function formatTimeAgo(dateString) {
        const now = new Date();
        const past = new Date(dateString);
        const diffInMinutes = Math.floor((now - past) / (1000 * 60));

        if (diffInMinutes < 1) return 'Just now';
        if (diffInMinutes < 60) return `${diffInMinutes} minutes ago`;
        
        const diffInHours = Math.floor(diffInMinutes / 60);
        if (diffInHours < 24) return `Last message ${diffInHours} hours ago`;
        
        const diffInDays = Math.floor(diffInHours / 24);
        if (diffInDays < 7) return `Last message ${diffInDays} days ago`;
        
        const diffInWeeks = Math.floor(diffInWeeks / 7);
        if (diffInWeeks < 4) return `Last message ${diffInWeeks} weeks ago`;
        
        return past.toLocaleDateString(); // Fallback for older chats
    }

    // Function to display messages in the main chat area
    function displayChatMessage(sender, message, isUser = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('chat-message');
        if (isUser) {
            messageDiv.classList.add('user');
        } else {
            messageDiv.classList.add('assistant');
        }
        messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
        conversationMessagesDiv.appendChild(messageDiv);
        conversationMessagesDiv.scrollTop = conversationMessagesDiv.scrollHeight; // Scroll to bottom
    }

    // Function to render chat sessions in the sidebar
    function renderChatSessions(sessionsToRender) {
        chatList.innerHTML = ''; // Clear existing list
        if (sessionsToRender.length === 0) {
            chatList.innerHTML = '<div style="padding: 20px; color: #6b7280; font-size: 14px; text-align: center;">No previous searches. Start a new one!</div>';
            newChatFloatingBtn.classList.add('visible'); // Show New Chat button if list is empty
        } else {
            newChatFloatingBtn.classList.remove('visible'); // Hide if chats exist
        }

        sessionsToRender.forEach(session => {
            const chatItem = document.createElement('div');
            chatItem.classList.add('chat-item');
            if (session.id === activeChatId) {
                chatItem.classList.add('active'); // Highlight active chat
            }
            chatItem.dataset.id = session.id; // Store unique ID
            chatItem.innerHTML = `
                <div class="chat-content-wrap">
                    <div class="chat-title">${session.title}</div>
                    <div class="chat-time">${session.timeAgo}</div>
                </div>
            `;
            chatItem.addEventListener('click', () => loadChatSession(session.id));
            chatList.appendChild(chatItem);
        });
    }

    // Function to load a specific chat session into the main chat area
    async function loadChatSession(id) {
        activeChatId = id;
        
        // Highlight active chat in sidebar
        document.querySelectorAll('.chat-item').forEach(item => {
            item.classList.remove('active');
            if (item.dataset.id === id) {
                item.classList.add('active');
            }
        });

        welcomeMessageDiv.classList.add('hidden'); // Hide welcome message
        conversationMessagesDiv.classList.add('active'); // Show conversation area
        conversationMessagesDiv.innerHTML = ''; // Clear current messages

        // Find the full conversation history for this session from the `allChatSessions` data
        const selectedSession = allChatSessions.find(session => session.id === id);
        if (selectedSession && selectedSession.messages) {
            currentActiveConversationMessages = selectedSession.messages.filter(msg => msg.role !== 'system');
            currentActiveConversationMessages.forEach(msg => {
                displayChatMessage(msg.role === 'user' ? 'You' : 'File Search System', msg.content, msg.role === 'user');
            });
            conversationMessagesDiv.scrollTop = conversationMessagesDiv.scrollHeight; // Scroll to bottom
        } else {
            displayChatMessage('System', 'No messages in this chat yet. Start typing!', false);
            currentActiveConversationMessages = [];
        }
    }


    // Function to fetch and update the chat sessions list for the sidebar
    async function fetchAllChatSessions() {
        try {
            const response = await fetch('chat_completions.json');
            if (!response.ok) {
                console.warn('chat_completions.json not found or could not be loaded. No chat history will be displayed.');
                allChatSessions = [];
                renderChatSessions([]); 
                return;
            }
            const fullHistory = await response.json();
            
            const sessionsMap = new Map(); 
            fullHistory.forEach(message => {
                if (message.role === 'system') return; 
                const sessionId = message.id; 
                if (!sessionId) { 
                    console.warn("Message without ID found, skipping:", message);
                    return;
                }

                if (!sessionsMap.has(sessionId)) {
                    sessionsMap.set(sessionId, {
                        id: sessionId,
                        title: '', 
                        messages: [],
                        lastMessageTime: null
                    });
                }
                const session = sessionsMap.get(sessionId);
                session.messages.push(message);
                session.lastMessageTime = message.timestamp || new Date().toISOString(); 
            });

            allChatSessions = Array.from(sessionsMap.values()).map(session => {
                session.timeAgo = formatTimeAgo(session.lastMessageTime);
                
                if (session.messages.length > 0) {
                     const firstUserMessage = session.messages.find(msg => msg.role === 'user');
                     const firstMessageContent = firstUserMessage ? firstUserMessage.content : session.messages[0].content;
                     session.title = firstMessageContent.substring(0, 50);
                     if (firstMessageContent.length > 50) session.title += '...';
                } else {
                    session.title = 'Empty Chat'; 
                }
                return session;
            });
            
            allChatSessions.sort((a, b) => new Date(b.lastMessageTime) - new Date(a.lastMessageTime));
            
            renderChatSessions(allChatSessions); // Render all fetched sessions

        } catch (error) {
            console.error('Error fetching chat sessions:', error);
            allChatSessions = [];
            renderChatSessions([]); 
        }
    }

    // --- Event Handlers ---

    // Handles sending messages to the backend and updating live chat display
    async function handleMessage(query) {
        // If no chat session is active (new chat), generate a new ID
        if (activeChatId === null) {
            activeChatId = `chat-${new Date().getTime()}`; 
            // Hide welcome message and show conversation area
            welcomeMessageDiv.classList.add('hidden');
            conversationMessagesDiv.classList.add('active');
            conversationMessagesDiv.innerHTML = ''; // Clear for new chat
        }

        displayChatMessage('You', query, true); 
        displayChatMessage('System', 'Processing...', false); 

        try {
            const response = await fetch('http://127.0.0.1:5000/search', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, chat_id: activeChatId }), 
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Network response was not ok.');
            }

            const data = await response.json();
            console.log("LLM Response:", data);

            // Remove "Processing..." message
            if (conversationMessagesDiv.lastChild && conversationMessagesDiv.lastChild.textContent.includes('Processing...')) {
                conversationMessagesDiv.removeChild(conversationMessagesDiv.lastChild);
            }

            if (data.result) {
                displayChatMessage('File Search System', data.result, false);
            } else {
                displayChatMessage('File Search System', 'No specific results found or an unexpected response occurred.', false);
            }

            // After a successful message, refresh sidebar to include the new interaction/session
            await fetchAllChatSessions();
            // Ensure the newly active/updated session is selected in the sidebar
            document.querySelectorAll('.chat-item').forEach(item => {
                item.classList.remove('active');
                if (item.dataset.id === activeChatId) {
                    item.classList.add('active');
                }
            });

        } catch (error) {
            console.error('Search failed:', error);
            if (conversationMessagesDiv.lastChild && conversationMessagesDiv.lastChild.textContent.includes('Processing...')) {
                conversationMessagesDiv.removeChild(conversationMessagesDiv.lastChild);
            }
            displayChatMessage('File Search System', `Error: ${error.message}. Please check your backend server and API key.`, false);
        }
    }

    // New Chat Floating Button functionality
    newChatFloatingBtn.addEventListener('click', async () => {
        activeChatId = null; // Clear active session to start new one
        currentActiveConversationMessages = []; // Reset local messages
        conversationMessagesDiv.innerHTML = ''; // Clear chat display
        welcomeMessageDiv.classList.remove('hidden'); // Show welcome message
        conversationMessagesDiv.classList.remove('active'); // Hide message area
        document.querySelectorAll('.chat-item').forEach(item => item.classList.remove('active')); // Deselect sidebar items

        // Also clear chat history on the backend
        try {
            const response = await fetch('http://127.0.0.1:5000/clear_chat', { method: 'POST' });
            if (!response.ok) {
                console.error('Failed to clear chat history on backend.');
            } else {
                console.log('Backend chat history cleared.');
                await fetchAllChatSessions(); // Refresh sidebar
            }
        } catch (error) {
            console.error('Error clearing backend chat:', error);
        }
        chatInput.focus(); // Focus input for new chat
    });

    // Sidebar search input for filtering chat history
    sidebarSearchInput.addEventListener('input', () => {
        const searchTerm = sidebarSearchInput.value.toLowerCase();
        const filteredSessions = allChatSessions.filter(session =>
            session.title.toLowerCase().includes(searchTerm) || 
            session.messages.some(msg => msg.content.toLowerCase().includes(searchTerm))
        );
        renderChatSessions(filteredSessions); // Render filtered sessions
    });

    // Send Message Button click handler
    sendMessageBtn.addEventListener('click', () => {
        const message = chatInput.value.trim();
        if (message) {
            handleMessage(message);
            chatInput.value = ''; // Clear input
            chatInput.style.height = 'auto'; // Reset textarea height
            sendMessageBtn.disabled = true; // Disable button
        }
    });

    // Auto-resize textarea and enable/disable send button
    window.autoResize = function(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = textarea.scrollHeight + 'px';
        sendMessageBtn.disabled = !textarea.value.trim(); // Enable/disable based on content
    };

    // Handle Enter key for sending messages (Shift+Enter for new line)
    window.handleKeyDown = function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault(); // Prevent new line
            sendMessageBtn.click(); // Trigger send button click
        }
    };

    // Close notification bar
    closeNotificationBtn.addEventListener('click', () => {
        notificationBar.style.display = 'none';
    });

    // Initial load of chat sessions and focus input
    fetchAllChatSessions();
    chatInput.focus();
});