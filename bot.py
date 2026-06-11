import os
import asyncio
import aiohttp
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ---------- CONFIGURATION ----------
OPENROUTER_API_KEY = "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"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# In-memory conversation history per user (user_id -> list of messages)
# Each message: {"role": "user" or "assistant", "content": "text"}
conversations = {}

# Max number of exchanges (user + assistant pairs) to keep per user
MAX_HISTORY_TURNS = 10

# ---------- HELPER: CALL OPENROUTER ----------
async def query_openrouter(messages):
    """Send messages to OpenRouter and return assistant's reply."""
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
        "Just send me any message and I'll reply. Use /clear to reset our conversation history.\n"
        "Source: https://openrouter.ai/"
    )

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Clear the conversation history for this user."""
    user_id = update.effective_user.id
    if user_id in conversations:
        del conversations[user_id]
    await update.message.reply_text("🧹 Conversation history cleared! Starting fresh.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle any text message: maintain history, call OpenRouter, reply."""
    user_id = update.effective_user.id
    user_text = update.message.text

    # Retrieve or initialize conversation history for this user
    if user_id not in conversations:
        # Start with an empty conversation (no system message)
        conversations[user_id] = []

    history = conversations[user_id]

    # Append user message
    history.append({"role": "user", "content": user_text})

    # Limit history to last MAX_HISTORY_TURNS pairs (each pair = user + assistant)
    # That's 2 * MAX_HISTORY_TURNS messages total.
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    # Send typing indicator
    await update.message.chat.send_action(action="typing")

    # Call OpenRouter with current history
    assistant_reply = await query_openrouter(history)

    # Append assistant reply to history
    history.append({"role": "assistant", "content": assistant_reply})

    # Again, trim if needed (should already be within limit, but double-check)
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    # Send the reply back to user
    await update.message.reply_text(assistant_reply)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors and optionally notify user."""
    print(f"Exception while handling an update: {context.error}")
    # Optionally send a message to the user
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "🤖 An internal error occurred. Please try again later or contact the bot admin."
        )

# ---------- MAIN ----------
def main():
    # Get token from environment variable
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("No TELEGRAM_BOT_TOKEN found in environment variables.")

    # Build the Application
    app = Application.builder().token(token).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("clear", clear))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register global error handler
    app.add_error_handler(error_handler)

    print("Bot is polling...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
