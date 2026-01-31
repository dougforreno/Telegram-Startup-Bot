"""
Reno Ecosystem Telegram Bot
Search for startup resources directly from Telegram.

Setup:
1. Create bot with @BotFather on Telegram
2. Get your bot token
3. Set environment variables
4. Deploy to Railway/Render/etc.

Usage:
    pip install python-telegram-bot supabase openai python-dotenv
    python telegram_bot.py
"""

import os
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from supabase import create_client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Optional: Restrict to specific users/groups
ALLOWED_USER_IDS = os.getenv("ALLOWED_USER_IDS", "").split(",")  # Comma-separated user IDs
ALLOWED_USER_IDS = [int(uid.strip()) for uid in ALLOWED_USER_IDS if uid.strip()]

# Initialize clients
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

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

def format_resource_short(r: dict, index: int) -> str:
    """Format a resource for the search results list."""
    cost_emoji = {"Free": "🆓", "Paid": "💰", "Varies": "💲"}.get(r.get("cost"), "")
    score = int(r.get("similarity", 0) * 100)
    
    return f"""*{index}. {r['name']}* {cost_emoji}
_{r.get('category', 'Resource')}_ • Match: {score}%
{r.get('description', '')[:150]}{'...' if len(r.get('description', '')) > 150 else ''}
"""

def format_resource_full(r: dict) -> str:
    """Format a resource with full details."""
    cost_emoji = {"Free": "🆓", "Paid": "💰", "Varies": "💲"}.get(r.get("cost_levels", {}).get("name") if r.get("cost_levels") else None, "")
    
    text = f"""📍 *{r['name']}* {cost_emoji}

📂 *Category:* {r.get('categories', {}).get('name', 'N/A') if r.get('categories') else 'N/A'}
🎯 *Stage:* {r.get('stages', {}).get('name', 'N/A') if r.get('stages') else 'N/A'}
💵 *Cost:* {r.get('cost_levels', {}).get('name', 'N/A') if r.get('cost_levels') else 'N/A'}
📍 *Geography:* {r.get('geographies', {}).get('name', 'N/A') if r.get('geographies') else 'N/A'}

📝 *Description:*
{r.get('description', 'No description available.')}
"""

    if r.get('best_for'):
        text += f"\n✅ *Best For:*\n{r['best_for']}\n"
    
    if r.get('intro_script'):
        text += f"\n💬 *How to Reach Out:*\n_{r['intro_script']}_\n"
    
    if r.get('website'):
        text += f"\n🔗 *Website:* {r['website']}\n"
    
    if r.get('wait_time'):
        text += f"\n⏱️ *Wait Time:* {r['wait_time']}\n"
    
    return text

# ============================================================================
# ACCESS CONTROL
# ============================================================================

def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    if not ALLOWED_USER_IDS:  # If no restrictions set, allow everyone
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
    
    welcome_text = f"""👋 Hi {user.first_name}! I'm the *Reno Startup Ecosystem Navigator*.

I can help you find resources for founders and entrepreneurs in the Reno-Tahoe region.

*How to use me:*
• Just type what you're looking for
• Be specific about your situation

*Example searches:*
• "veteran starting a business"
• "free mentorship for early stage startups"
• "funding for tech companies"
• "workspace in Reno"

*Commands:*
/search <query> - Search for resources
/categories - List all categories
/help - Show this message

Just type your question and I'll find the best matches! 🚀
"""
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    await start(update, context)

async def categories_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List all categories."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    result = supabase.table("categories").select("name").order("display_order").execute()
    categories = [c["name"] for c in result.data]
    
    text = "*📂 Resource Categories:*\n\n"
    for cat in categories:
        text += f"• {cat}\n"
    
    text += "\n_Search any category by typing it!_"
    await update.message.reply_text(text, parse_mode='Markdown')

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    if not context.args:
        await update.message.reply_text("Please provide a search query.\n\nExample: `/search veteran business help`", parse_mode='Markdown')
        return
    
    query = " ".join(context.args)
    await perform_search(update, query)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle regular text messages as search queries."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    query = update.message.text
    await perform_search(update, query)

async def perform_search(update: Update, query: str) -> None:
    """Perform search and send results."""
    # Send "searching" message
    searching_msg = await update.message.reply_text("🔍 Searching...")
    
    try:
        # Search
        results = search_resources(query, limit=5)
        
        if not results:
            await searching_msg.edit_text(
                f"😕 No resources found for: *{query}*\n\nTry a different search term.",
                parse_mode='Markdown'
            )
            return
        
        # Format results
        text = f"🔍 *Results for:* _{query}_\n\n"
        
        # Create inline keyboard for "More Details" buttons
        keyboard = []
        
        for i, r in enumerate(results, 1):
            text += format_resource_short(r, i)
            text += "\n"
            
            # Add button for each result
            keyboard.append([
                InlineKeyboardButton(
                    f"📋 Details: {r['name'][:30]}...",
                    callback_data=f"details_{r['id']}"
                )
            ])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await searching_msg.edit_text(
            text,
            parse_mode='Markdown',
            reply_markup=reply_markup,
            disable_web_page_preview=True
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        await searching_msg.edit_text(
            "❌ Sorry, an error occurred. Please try again.",
            parse_mode='Markdown'
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if not is_authorized(query.from_user.id):
        await query.edit_message_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    data = query.data
    
    if data.startswith("details_"):
        resource_id = int(data.replace("details_", ""))
        
        try:
            resource = get_resource_details(resource_id)
            
            if resource:
                text = format_resource_full(resource)
                
                # Add "Back to search" button
                keyboard = [[
                    InlineKeyboardButton("🔙 New Search", callback_data="new_search")
                ]]
                
                if resource.get('website'):
                    keyboard[0].insert(0, 
                        InlineKeyboardButton("🌐 Website", url=resource['website'])
                    )
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    text,
                    parse_mode='Markdown',
                    reply_markup=reply_markup,
                    disable_web_page_preview=True
                )
            else:
                await query.edit_message_text("❌ Resource not found.")
                
        except Exception as e:
            logger.error(f"Details error: {e}")
            await query.edit_message_text("❌ Error loading details.")
    
    elif data == "new_search":
        await query.edit_message_text(
            "🔍 *Ready for a new search!*\n\nJust type what you're looking for.",
            parse_mode='Markdown'
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Update {update} caused error {context.error}")

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set")
        return
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE credentials not set")
        return
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set")
        return
    
    print("🤖 Starting Reno Ecosystem Telegram Bot...")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("categories", categories_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start polling
    print("✅ Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
