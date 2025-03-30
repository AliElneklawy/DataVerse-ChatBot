Bot Implementations
===================

DataVerse ChatBot can be deployed as messaging bots for Telegram and WhatsApp platforms. This module documents these bot implementations.

Telegram Bot
------------

The Telegram bot implementation uses the ``python-telegram-bot`` library to create a conversational interface.

Key Commands:
- ``/start``: Initiates conversation with the bot
- ``/help``: Displays help information
- ``/reset``: Clears conversation history

Key Functions:
- ``start(update, context)``: Handles the /start command
- ``help_command(update, context)``: Handles the /help command
- ``reset(update, context)``: Handles the /reset command to clear conversation history
- ``upload_file(update, context)``: Handles file uploads to incorporate into the knowledge base
- ``handle_message(update, context)``: Processes text messages from users
- ``handle_voice(update, context)``: Processes voice messages from users
- ``error_handler(update, context)``: Handles errors in the Telegram bot

WhatsApp Bot
------------

The WhatsApp bot implementation uses Twilio's API to integrate with WhatsApp messaging.

Key Functions:
- ``handle_message()``: Processes incoming WhatsApp messages
- ``text_message_handler(from_number, message_body)``: Handles text messages from WhatsApp
- ``voice_message_handler(from_number, media_url)``: Handles voice messages from WhatsApp
- ``file_handler(from_number, media_url, content_type)``: Handles file uploads from WhatsApp

Common Bot Features
-------------------

Both bot implementations share common features:

1. **Text Conversation**: Process text messages and generate responses using the RAG system
2. **Voice Interaction**: Transcribe voice messages and respond in text
3. **File Processing**: Accept file uploads to expand the knowledge base
4. **History Management**: Track conversation history per user
5. **Command Handling**: Support for commands like /start, /help, and /reset

Implementation Details
----------------------

Authentication and Security
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both bot implementations use token-based authentication:

- **Telegram**: Uses a bot token from the BotFather
- **WhatsApp/Twilio**: Uses account SID and auth token from Twilio

These tokens should be stored in the `.env` file:

.. code-block:: ini

   # Telegram Bot Token
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   
   # Twilio (WhatsApp)
   TWILIO_ACCOUNT_SID=your_twilio_account_sid
   TWILIO_AUTH_TOKEN=your_twilio_auth_token
   TWILIO_PHONE_NUMBER=your_twilio_phone_number

User Identification
~~~~~~~~~~~~~~~~~~~

Each bot platform handles user identification differently:

- **Telegram**: Uses the user's Telegram ID as the unique identifier
- **WhatsApp**: Uses the user's WhatsApp phone number as the unique identifier

These identifiers are used to maintain chat history and track usage metrics.

Deployment Recommendations
--------------------------

When deploying bots to production:

1. Use a secure hosting environment with HTTPS
2. Implement rate limiting to prevent abuse
3. Set up monitoring for bot availability
4. Create backup and recovery procedures
5. Consider privacy and data retention policies for user messages