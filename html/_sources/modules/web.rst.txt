Web Interfaces
==============

The ``web`` module provides web-based interfaces for the DataVerse ChatBot, including:

- Chat web application with iframe embedding
- Admin dashboard for system management

Chat Web Application
--------------------

The chat web application provides a FastAPI-based interface for embedding the chatbot in websites via an iframe.

Key Features:
- Real-time chat interface
- Message history display
- Voice input support
- Customizable appearance
- Responsive design for mobile and desktop
- Iframe embedding capability

Main Functions:
- ``home()``: Serves the iframe HTML interface
- ``chat(request)``: Handles chat requests from the web interface
- ``transcribe_audio(file)``: Processes audio transcription requests

Chat Web Template
-----------------

This module contains the HTML template for the chat interface.

The ``IFRAME_HTML`` variable provides the HTML template for embedding the chat interface in an iframe.

Admin Dashboard
---------------

The admin dashboard provides tools for managing the RAG system. It uses Dash and includes features to:

- View system metrics
- Monitor active conversations
- Track token usage and costs
- Manage content (crawl websites, upload files)
- Update account settings

Key Functions:
- ``serve_layout()``: Creates the dashboard layout
- ``register_callbacks(app)``: Registers all dashboard callbacks

Main Dashboard Pages
~~~~~~~~~~~~~~~~~~~~

The admin dashboard consists of several pages:

1. **Dashboard**: Overview of system metrics and recent activity
2. **Content Management**: Tools to add and manage content sources
3. **User Management**: View and manage system users
4. **Account Settings**: Update admin credentials
5. **System Settings**: Configure system parameters
6. **Chat History**: View and search chat interactions

Authentication System
~~~~~~~~~~~~~~~~~~~~~

The dashboard includes authentication features:
- ``authenticate_user(username, password)``: Authenticates users against stored credentials
- ``update_credentials(username, password)``: Updates admin credentials

Content Management
~~~~~~~~~~~~~~~~~~

Content management capabilities include:
- ``crawl_website(url, max_depth, client)``: Crawls websites to add content to the RAG system
- ``upload_file(content, filename)``: Uploads and processes files for RAG

Metrics and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

The dashboard provides visual analytics:
- ``generate_metrics()``: Generates system metrics for the dashboard
- ``create_usage_chart()``: Creates a chart of model usage distribution
- ``create_cost_chart()``: Creates a chart of cost distribution by model

Style and Appearance
~~~~~~~~~~~~~~~~~~~~

The dashboard uses custom CSS for styling, stored in:
- ``static/css/admin.css``: Main dashboard styling
- ``static/css/dark_mode.css``: Dark mode styling

JavaScript functionality is provided in:
- ``static/js/admin.js``: Dashboard interactivity