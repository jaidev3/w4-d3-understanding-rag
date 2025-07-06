# Discord MCP Server

## Overview
This is a Model Context Protocol (MCP) server for Discord integration using FastMCP. It provides tools for interacting with Discord channels, sending messages, retrieving message history, and moderating content.

## Prerequisites
- Python 3.8+
- Discord Bot Token
- Discord Bot with appropriate permissions

## Setup

### 1. Create a Discord Bot
1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to the "Bot" section and create a bot
4. Copy the bot token (you'll need this later)
5. Under "Privileged Gateway Intents", enable:
   - Presence Intent
   - Server Members Intent
   - Message Content Intent

### 2. Install Dependencies
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root and add your Discord bot token:
```
DISCORD_BOT_TOKEN=your_discord_bot_token_here
```

### 4. Invite Bot to Server
1. Go to the OAuth2 section in your Discord application
2. Select "bot" scope
3. Select required permissions:
   - Send Messages
   - Read Message History
   - Manage Messages (for moderation)
   - Timeout Members (for moderation)
4. Use the generated URL to invite the bot to your server

## Running the Server

### Start the MCP Server
```bash
python discord_mcp_server.py
```

The server will start and be available for MCP connections. The FastMCP server will handle the MCP protocol communication.

### Server Information
- The MCP server runs using FastMCP
- Default configuration handles MCP protocol automatically
- Server provides tools for Discord interaction via MCP

## Available Tools

### 1. `send_message(channel_id: int, message: str)`
Send a message to a specific Discord channel.
- **channel_id**: The ID of the Discord channel
- **message**: The message content to send
- **Returns**: Success confirmation message

### 2. `get_messages(channel_id: int, limit: int = 10)`
Retrieve message history from a Discord channel.
- **channel_id**: The ID of the Discord channel
- **limit**: Number of messages to retrieve (default: 10)
- **Returns**: List of message contents

### 3. `get_channel_info(channel_id: int)`
Fetch metadata about a Discord channel.
- **channel_id**: The ID of the Discord channel
- **Returns**: Channel metadata including name, type, topic, etc.

### 4. `search_messages(channel_id: int, query: str, limit: int = 10)`
Search for messages in a Discord channel.
- **channel_id**: The ID of the Discord channel to search
- **query**: The search query string
- **limit**: Number of messages to search through (default: 10)
- **Returns**: List of matching message contents

### 5. `moderate_content(channel_id: int, message_id: int, action: str)`
Moderate content in a Discord channel.
- **channel_id**: The ID of the Discord channel
- **message_id**: The ID of the message to moderate
- **action**: The moderation action ("delete", "warn", "mute")
- **Returns**: Success confirmation message

## Getting Channel and Message IDs

### Channel ID
1. Enable Developer Mode in Discord (User Settings > Advanced > Developer Mode)
2. Right-click on a channel and select "Copy ID"

### Message ID
1. Right-click on a message and select "Copy ID"

## Troubleshooting

### Common Issues
1. **Bot not responding**: Ensure the bot token is correct and the bot is online
2. **Permission errors**: Verify the bot has necessary permissions in the target channels
3. **Channel not found**: Double-check the channel ID and ensure the bot has access

### Debug Mode
For debugging, you can uncomment the actual Discord API calls in the code (currently commented out for safety).

## Notes
- The server uses FastMCP for MCP protocol handling
- All Discord operations are currently stubbed out for safety - uncomment the actual Discord API calls when ready to use
- Ensure your bot has appropriate permissions for the operations you want to perform
- The bot requires all intents to be enabled for full functionality 