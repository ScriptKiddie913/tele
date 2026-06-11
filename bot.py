"""
Telegram bot that uses OpenRouter API for AI chat.
Reads all configuration from environment variables.
"""

import os
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------- CONFIGURATION FROM ENVIRONMENT ----------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Missing TELEGRAM_BOT_TOKEN environment variable")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY environment variable")

# Model – can be overridden with OPENROUTER_MODEL env var
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

# Max conversation turns (user + assistant pairs) to keep
try:
    MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
except ValueError:
    MAX_HISTORY_TURNS = 10

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# In-memory conversation history per user
# Format: { user_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...] }
conversations = {}

# ---------- HELPER: CALL OPENROUTER API ----------
async def query_openrouter(messages):
    """
    Send a list of messages to OpenRouter and return the assistant's reply.
    Handles timeouts and HTTP errors gracefully.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
    }
    timeout = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(OPENROUTER_URL, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Extract the assistant's reply
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await resp.text()
                    return f"⚠️ OpenRouter error (HTTP {resp.status}): {error_text[:200]}"
        except asyncio.TimeoutError:
            return "⏰ Request timed out. Please try again later."
        except Exception as e:
            return f"❌ Unexpected error: {str(e)[:200]}"

# ---------- TELEGRAM HANDLERS ----------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hello {user.first_name}!\n"
        f"I'm a chat bot powered by OpenRouter (model: {MODEL}).\n"
        "Just send me any message and I'll reply.\n"
        "Commands:\n"
        "/clear – reset our conversation history\n"
        "/status – show current model and history limit\n"
        "Source: https://openrouter.ai/"
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the conversation history for this user."""
    user_id = update.effective_user.id
    if user_id in conversations:
        del conversations[user_id]
    await update.message.reply_text("🧹 Conversation history cleared! Starting fresh.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show current configuration and conversation length."""
    user_id = update.effective_user.id
    history_len = len(conversations.get(user_id, []))
    await update.message.reply_text(
        f"🤖 Bot status:\n"
        f"Model: `{MODEL}`\n"
        f"Max history turns: {MAX_HISTORY_TURNS}\n"
        f"Messages in current session: {history_len}\n"
        f"(each turn = user + assistant message)"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle any text message (not a command):
    - Maintain per-user conversation history
    - Call OpenRouter
    - Store and send back the reply
    """
    user_id = update.effective_user.id
    user_text = update.message.text

    # Initialize conversation history for this user if not present
    if user_id not in conversations:
        conversations[user_id] = []

    history = conversations[user_id]

    # Append the user's new message
    history.append({"role": "user", "content": user_text})

    # Trim history to the last (MAX_HISTORY_TURNS * 2) messages
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    # Send typing indicator to let user know the bot is processing
    await update.message.chat.send_action(action="typing")

    # Call OpenRouter with the current history
    assistant_reply = await query_openrouter(history)

    # Append assistant reply to history
    history.append({"role": "assistant", "content": assistant_reply})

    # Trim again just in case (should already be within limit)
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    # Send the reply back to the user
    await update.message.reply_text(assistant_reply)

# ---------- GLOBAL ERROR HANDLER ----------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and optionally notify the user."""
    print(f"Exception while handling an update: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "🤖 An internal error occurred. Please try again later or contact the bot admin."
        )

# ---------- MAIN FUNCTION ----------
def main():
    """Start the bot."""
    # Build the Application
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(CommandHandler("status", status))

    # Add message handler for all text messages (excluding commands)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register global error handler
    app.add_error_handler(error_handler)

    print(f"Bot is polling... using model: {MODEL}")
    print(f"Max history turns: {MAX_HISTORY_TURNS}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
