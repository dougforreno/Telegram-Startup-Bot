"""
Reno Ecosystem Telegram Bot with Claude AI
Natural conversation-powered search for startup resources.

Features:
- Claude AI for intelligent, conversational responses
- Semantic search of resource database
- Context-aware follow-up questions
- Personalized recommendations

Setup:
1. Create bot with @BotFather on Telegram
2. Get your bot token
3. Set environment variables
4. Deploy to Railway/Render/etc.

Usage:
    pip install python-telegram-bot anthropic supabase openai python-dotenv
    python telegram_bot.py
"""

import os
import logging
import json
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from supabase import create_client
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Optional: Restrict to specific users/groups
ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "").split(",")
ALLOWED_USER_IDS = [int(uid.strip()) for uid in ALLOWED_USER_IDS if uid.strip()]

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Conversation history storage (in production, use Redis or database)
conversation_history = {}

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================================================
# SYSTEM PROMPT FOR CLAUDE
# ============================================================================

SYSTEM_PROMPT = """You are the Reno Startup Ecosystem Navigator, a helpful assistant that connects founders with the right resources in the Reno-Tahoe region.

You have access to a database of startup resources including mentorship programs, funding opportunities, accelerators, workspaces, and government resources.

## YOUR PERSONALITY
- Friendly, encouraging, and supportive
- Concise (this is Telegram - keep responses brief!)
- Practical and action-oriented
- Knowledgeable about the startup journey

## HOW TO RESPOND

1. **When search results are provided:**
   - Explain WHY each resource is relevant to their specific situation
   - Highlight the most important 2-3 resources
   - Include key details: cost (free/paid), what they offer
   - Suggest how to reach out if intro_script is available
   - Ask a follow-up question to help narrow down or expand the search

2. **When NO search results are provided:**
   - Ask clarifying questions to understand their needs
   - Try to understand: their stage, what help they need, any relevant background

3. **For follow-up questions:**
   - Remember context from the conversation
   - Offer to search for different resources if needed
   - Provide practical next steps

## FORMATTING FOR TELEGRAM
- Use *bold* for resource names and key points
- Use _italics_ for categories or emphasis
- Keep paragraphs short (2-3 sentences max)
- Use emojis sparingly but effectively: 🚀 💡 ✅ 🔗 💰 🆓
- Don't use markdown headers (##) - they don't render in Telegram

## EXAMPLE RESPONSE STYLE

"Based on what you shared, here are my top picks:

*1. Nevada SBDC* 🆓
Perfect for early-stage founders who need help with business planning. They offer free one-on-one advising.

*2. StartUpNV* 
If you're building a scalable tech company, their accelerator could be a great fit.

When reaching out to SBDC, try: _"I'm starting a business and would like help creating a business plan."_

Would you like more details on either of these, or should I look for something more specific like funding or workspace?"
"""

# ============================================================================
# SEARCH FUNCTIONS
# ============================================================================

def generate_embedding(text: str) -> list[float]:
    """Generate embedding for search query."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_resources(query: str, limit: int = 5) -> list[dict]:
    """Search resources using vector similarity."""
    embedding = generate_embedding(query)
    
    result = supabase.rpc("search_resources", {
        "query_embedding": embedding,
        "community_slug": "reno",
        "match_count": limit,
        "similarity_threshold": 0.3
    }).execute()
    
    return result.data

def get_resource_details(resource_id: int) -> dict:
    """Get full details for a resource."""
    result = supabase.table("resources").select(
        "*, categories(name), stages(name), cost_levels(name), geographies(name)"
    ).eq("id", resource_id).single().execute()
    return result.data

def get_resource_tags(resource_id: int) -> dict:
    """Get service and audience tags for a resource."""
    services = supabase.table("resource_services").select(
        "service_tags(name)"
    ).eq("resource_id", resource_id).execute()
    
    audiences = supabase.table("resource_audiences").select(
        "audience_tags(name)"
    ).eq("resource_id", resource_id).execute()
    
    return {
        "services": [s["service_tags"]["name"] for s in services.data if s.get("service_tags")],
        "audiences": [a["audience_tags"]["name"] for a in audiences.data if a.get("audience_tags")]
    }

def format_resources_for_claude(resources: list[dict]) -> str:
    """Format search results for Claude to understand."""
    if not resources:
        return "NO RESULTS FOUND"
    
    formatted = []
    for r in resources:
        tags = get_resource_tags(r["id"])
        
        resource_text = f"""
RESOURCE: {r['name']}
- ID: {r['id']}
- Category: {r.get('category', 'N/A')}
- Stage: {r.get('stage', 'N/A')}
- Cost: {r.get('cost', 'N/A')}
- Geography: {r.get('geography', 'N/A')}
- Description: {r.get('description', 'N/A')}
- Best For: {r.get('best_for', 'N/A')}
- Not Good For: {r.get('not_good_for', 'N/A')}
- Intro Script: {r.get('intro_script', 'N/A')}
- Website: {r.get('website', 'N/A')}
- Wait Time: {r.get('wait_time', 'N/A')}
- Service Tags: {', '.join(tags['services']) if tags['services'] else 'N/A'}
- Audience Tags: {', '.join(tags['audiences']) if tags['audiences'] else 'N/A'}
- Relevance Score: {int(r.get('similarity', 0) * 100)}%
"""
        formatted.append(resource_text)
    
    return "\n---\n".join(formatted)

# ============================================================================
# CLAUDE CONVERSATION
# ============================================================================

def get_conversation_history(user_id: int) -> list[dict]:
    """Get conversation history for a user."""
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    return conversation_history[user_id]

def add_to_history(user_id: int, role: str, content: str):
    """Add message to conversation history."""
    history = get_conversation_history(user_id)
    history.append({"role": role, "content": content})
    
    # Keep only last 10 messages to manage context length
    if len(history) > 10:
        conversation_history[user_id] = history[-10:]

def clear_history(user_id: int):
    """Clear conversation history for a user."""
    conversation_history[user_id] = []

def should_search(message: str, history: list[dict]) -> tuple[bool, str]:
    """Use Claude to determine if we should search and what query to use."""
    
    # Quick heuristics first
    search_indicators = [
        "looking for", "need help", "find", "search", "recommend",
        "resources", "funding", "mentor", "accelerator", "workspace",
        "starting", "business", "startup", "entrepreneur", "founder",
        "veteran", "student", "woman", "grant", "investor"
    ]
    
    message_lower = message.lower()
    
    # If it's clearly a search request
    if any(indicator in message_lower for indicator in search_indicators):
        return True, message
    
    # If it's a short follow-up like "yes", "tell me more", etc.
    short_responses = ["yes", "yeah", "sure", "ok", "okay", "tell me more", "more", "details"]
    if message_lower.strip() in short_responses:
        return False, ""
    
    # For ambiguous cases, let Claude decide
    return True, message

async def get_claude_response(user_id: int, user_message: str, search_results: str = None) -> str:
    """Get response from Claude with optional search results."""
    
    history = get_conversation_history(user_id)
    
    # Build the user message with search context if available
    if search_results and search_results != "NO RESULTS FOUND":
        augmented_message = f"""USER MESSAGE: {user_message}

SEARCH RESULTS FROM DATABASE:
{search_results}

Please analyze these results and provide a helpful, personalized response. Explain why these resources are relevant to their situation."""
    elif search_results == "NO RESULTS FOUND":
        augmented_message = f"""USER MESSAGE: {user_message}

No resources were found matching this query. Please ask clarifying questions to better understand what they need, or suggest they try different search terms."""
    else:
        augmented_message = user_message
    
    # Add user message to history
    messages = history + [{"role": "user", "content": augmented_message}]
    
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages
        )
        
        assistant_message = response.content[0].text
        
        # Store the original user message (not the augmented one) in history
        add_to_history(user_id, "user", user_message)
        add_to_history(user_id, "assistant", assistant_message)
        
        return assistant_message
        
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return "I'm having trouble connecting right now. Please try again in a moment."

# ============================================================================
# ACCESS CONTROL
# ============================================================================

def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS

# ============================================================================
# BOT HANDLERS
# ============================================================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message when /start is issued."""
    user = update.effective_user
    
    if not is_authorized(user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    # Clear conversation history for fresh start
    clear_history(user.id)
    
    welcome_text = f"""👋 Hi {user.first_name}! I'm the *Reno Startup Ecosystem Navigator*.

I can help you find the right resources for your entrepreneurial journey in the Reno-Tahoe region.

Just tell me about yourself and what you're looking for. For example:

• _"I'm a veteran looking to start a small business"_
• _"I need funding for my tech startup"_
• _"Where can I find free mentorship?"_

What can I help you with today? 🚀"""

    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    help_text = """*How to use this bot:*

Just chat naturally! Tell me:
• What stage you're at (idea, early, growing)
• What kind of help you need
• Any relevant background (veteran, student, etc.)

*Commands:*
/start - Start fresh conversation
/clear - Clear conversation history
/help - Show this message

*Tips:*
• Be specific about what you need
• Ask follow-up questions
• I remember our conversation context!"""

    await update.message.reply_text(help_text, parse_mode='Markdown')

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    clear_history(update.effective_user.id)
    await update.message.reply_text("🔄 Conversation cleared! What would you like to explore?")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages with Claude AI."""
    user = update.effective_user
    
    if not is_authorized(user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    user_message = update.message.text
    
    # Show typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Determine if we should search
        should_do_search, search_query = should_search(user_message, get_conversation_history(user.id))
        
        search_results = None
        resource_buttons = []
        
        if should_do_search:
            # Search the database
            results = search_resources(search_query, limit=5)
            
            if results:
                search_results = format_resources_for_claude(results)
                
                # Create buttons for top results
                for r in results[:3]:  # Top 3 as buttons
                    resource_buttons.append([
                        InlineKeyboardButton(
                            f"📋 {r['name'][:35]}{'...' if len(r['name']) > 35 else ''}",
                            callback_data=f"details_{r['id']}"
                        )
                    ])
            else:
                search_results = "NO RESULTS FOUND"
        
        # Get Claude's response
        response = await get_claude_response(user.id, user_message, search_results)
        
        # Send response with optional resource buttons
        if resource_buttons:
            reply_markup = InlineKeyboardMarkup(resource_buttons)
            await update.message.reply_text(
                response,
                parse_mode='Markdown',
                reply_markup=reply_markup,
                disable_web_page_preview=True
            )
        else:
            await update.message.reply_text(
                response,
                parse_mode='Markdown',
                disable_web_page_preview=True
            )
            
    except Exception as e:
        logger.error(f"Message handling error: {e}")
        await update.message.reply_text(
            "Sorry, I encountered an error. Please try again or use /start to restart."
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks for resource details."""
    query = update.callback_query
    await query.answer()
    
    if not is_authorized(query.from_user.id):
        return
    
    data = query.data
    
    if data.startswith("details_"):
        resource_id = int(data.replace("details_", ""))
        
        try:
            resource = get_resource_details(resource_id)
            
            if resource:
                # Format detailed view
                cost = resource.get('cost_levels', {}).get('name', 'N/A') if resource.get('cost_levels') else 'N/A'
                cost_emoji = {"Free": "🆓", "Paid": "💰", "Varies": "💲"}.get(cost, "")
                
                text = f"""📍 *{resource['name']}* {cost_emoji}

*Category:* {resource.get('categories', {}).get('name', 'N/A') if resource.get('categories') else 'N/A'}
*Stage:* {resource.get('stages', {}).get('name', 'N/A') if resource.get('stages') else 'N/A'}
*Cost:* {cost}

{resource.get('description', 'No description available.')}
"""

                if resource.get('best_for'):
                    text += f"\n✅ *Best For:*\n{resource['best_for']}\n"
                
                if resource.get('intro_script'):
                    text += f"\n💬 *How to reach out:*\n_{resource['intro_script']}_\n"
                
                # Buttons
                keyboard = []
                if resource.get('website'):
                    keyboard.append([InlineKeyboardButton("🌐 Visit Website", url=resource['website'])])
                
                reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
                
                await query.message.reply_text(
                    text,
                    parse_mode='Markdown',
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
            else:
                await query.message.reply_text("❌ Resource not found.")
                
        except Exception as e:
            logger.error(f"Details error: {e}")
            await query.message.reply_text("❌ Error loading details.")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Update {update} caused error {context.error}")

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Start the bot."""
    # Validate configuration
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not SUPABASE_URL or not SUPABASE_KEY:
        missing.append("SUPABASE_URL/SUPABASE_KEY")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY:
        missing.append("ANTHROPIC_API_KEY")
    
    if missing:
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        return
    
    print("🤖 Starting Reno Ecosystem Telegram Bot (with Claude AI)...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start polling
    print("✅ Bot is running with Claude AI! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
