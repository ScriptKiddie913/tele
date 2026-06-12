import os
import asyncio
import logging
from typing import Dict, List

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load environment variables
load_dotenv()

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-3.5-turbo"  # You can change this to any OpenRouter supported model
MAX_CONVERSATION_LEN = 10  # Number of message pairs to remember per user
SYSTEM_PROMPT = "You are a helpful assistant."

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Store conversation history in memory (for demo; use persistent DB in production)
conversations: Dict[int, List[Dict[str, str]]] = {}

def get_conversation(chat_id: int) -> List[Dict[str, str]]:
    """Get conversation history for a chat_id, initializing if needed."""
    if chat_id not in conversations:
        conversations[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return conversations[chat_id]

def trim_conversation(messages: List[Dict[str, str]], max_pairs: int) -> List[Dict[str, str]]:
    """
    Keep system message + last N user-assistant pairs.
    Each pair = 2 messages (user + assistant). System message excluded from count.
    """
    system_messages = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    
    # Keep last `max_pairs * 2` non-system messages
    max_non_system = max_pairs * 2
    if len(non_system) > max_non_system:
        non_system = non_system[-max_non_system:]
    
    return system_messages + non_system

async def query_openrouter(messages: List[Dict[str, str]]) -> str:
    """Send messages to OpenRouter and return assistant's reply."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            return f"Error from OpenRouter: HTTP {e.response.status_code}"
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            return f"Network error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return "An unexpected error occurred."

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm an AI bot powered by OpenRouter.\n"
        f"Send me any message and I'll reply using the {MODEL} model.\n"
        f"Commands:\n/start - Show this message\n/reset - Clear conversation history\n/help - Show help"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    await update.message.reply_text(
        "Just send me a message and I'll respond!\n"
        "I remember the last few messages per conversation (up to {MAX_CONVERSATION_LEN} exchanges).\n"
        "Use /reset to clear the conversation history."
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset conversation history for the user."""
    chat_id = update.effective_chat.id
    conversations[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    await update.message.reply_text("Conversation history cleared! Starting fresh.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user message: add to history, get AI response, reply."""
    chat_id = update.effective_chat.id
    user_message = update.message.text
    
    # Ignore empty messages
    if not user_message:
        return
    
    # Show typing indicator
    await update.message.chat.send_action(action="typing")
    
    # Get conversation history
    conversation = get_conversation(chat_id)
    
    # Add user message
    conversation.append({"role": "user", "content": user_message})
    
    # Trim history if needed
    conversation = trim_conversation(conversation, MAX_CONVERSATION_LEN)
    
    # Get AI response
    ai_response = await query_openrouter(conversation)
    
    # Add assistant's reply to history (only if no error occurred that broke the format)
    if not ai_response.startswith("Error") and not ai_response.startswith("Network"):
        conversation.append({"role": "assistant", "content": ai_response})
    
    # Update stored conversation (after potential trimming again)
    conversations[chat_id] = trim_conversation(conversation, MAX_CONVERSATION_LEN)
    
    # Send response (split if too long for Telegram)
    if len(ai_response) > 4096:
        for i in range(0, len(ai_response), 4096):
            await update.message.reply_text(ai_response[i:i+4096])
    else:
        await update.message.reply_text(ai_response)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.warning(f"Update {update} caused error {context.error}")

def main() -> None:
    """Start the bot."""
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN environment variable not set")
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Create application
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    # Start polling
    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
