"""
Reno Ecosystem Telegram Bot with Claude AI - Multi-Language
Natural conversation-powered search for startup resources.
Automatically responds in the user's language.

Features:
- Claude AI for intelligent, conversational responses
- Automatic language detection and response
- Supports English, Spanish, Chinese, Tagalog, and more
- Semantic search of resource database
- Context-aware follow-up questions

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
# SUPPORTED LANGUAGES
# ============================================================================

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish (Español)",
    "zh": "Chinese (中文)",
    "tl": "Tagalog (Filipino)",
    "vi": "Vietnamese (Tiếng Việt)",
    "ko": "Korean (한국어)",
    "ja": "Japanese (日本語)",
    "ar": "Arabic (العربية)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "pt": "Portuguese (Português)",
    "hi": "Hindi (हिन्दी)",
    "ru": "Russian (Русский)",
}

# ============================================================================
# SYSTEM PROMPT FOR CLAUDE (Multi-language)
# ============================================================================

SYSTEM_PROMPT = """You are the Reno Startup Ecosystem Navigator, a helpful assistant that connects founders with the right resources in the Reno-Tahoe region.

## CRITICAL: LANGUAGE HANDLING
- ALWAYS detect the language of the user's message
- ALWAYS respond in the SAME language the user writes in
- If they write in Spanish, respond entirely in Spanish
- If they write in Chinese, respond entirely in Chinese
- If they mix languages, respond in their primary language
- Resource names can stay in English (they're proper nouns), but your explanations must be in the user's language

## LANGUAGE EXAMPLES

English: "I need help starting a business"
→ Respond in English

Spanish: "Necesito ayuda para empezar un negocio"
→ Respond entirely in Spanish: "¡Excelente! Aquí hay recursos que te pueden ayudar..."

Chinese: "我需要帮助开始创业"
→ Respond entirely in Chinese: "太好了！以下是一些可以帮助您的资源..."

Tagalog: "Kailangan ko ng tulong sa pagsisimula ng negosyo"
→ Respond entirely in Tagalog: "Maganda! Narito ang mga resources na makakatulong sa iyo..."

Vietnamese: "Tôi cần giúp đỡ để bắt đầu kinh doanh"
→ Respond entirely in Vietnamese: "Tuyệt vời! Đây là những nguồn lực có thể giúp bạn..."

## YOUR PERSONALITY
- Friendly, encouraging, and supportive
- Concise (this is Telegram - keep responses reasonably brief)
- Practical and action-oriented
- Culturally aware and respectful

## HOW TO RESPOND

1. **When search results are provided:**
   - Explain WHY each resource is relevant to their situation (in their language)
   - Highlight the most important 2-3 resources
   - Include key details: cost (free/paid), what they offer
   - Suggest how to reach out if intro_script is available
   - Ask a follow-up question in their language

2. **When NO search results are provided:**
   - Ask clarifying questions (in their language)
   - Try to understand: their stage, what help they need, any relevant background

3. **For follow-up questions:**
   - Remember context from the conversation
   - Continue in the same language they've been using

## FORMATTING FOR TELEGRAM
- Use *bold* for resource names and key points
- Use _italics_ for emphasis
- Keep paragraphs short (2-3 sentences max)
- Use emojis appropriately: 🚀 💡 ✅ 🔗 💰 🆓
- Don't use markdown headers (##)

## SPECIAL PHRASES BY LANGUAGE

Free = Gratis (ES) = 免费 (ZH) = Libre (TL) = Miễn phí (VI) = 무료 (KO) = 無料 (JA)
Paid = De pago (ES) = 付费 (ZH) = Bayad (TL) = Trả phí (VI) = 유료 (KO) = 有料 (JA)
Website = Sitio web (ES) = 网站 (ZH) = Website (TL) = Trang web (VI) = 웹사이트 (KO) = ウェブサイト (JA)
"""

# ============================================================================
# WELCOME MESSAGES BY LANGUAGE
# ============================================================================

WELCOME_MESSAGES = {
    "en": """👋 Hi {name}! I'm the *Reno Startup Ecosystem Navigator*.

I help founders find resources in the Reno-Tahoe region. I speak multiple languages - just write to me in yours!

Tell me about yourself and what you're looking for. For example:
• _"I'm a veteran looking to start a small business"_
• _"I need funding for my tech startup"_
• _"Where can I find free mentorship?"_

What can I help you with today? 🚀""",

    "es": """👋 ¡Hola {name}! Soy el *Navegador del Ecosistema Startup de Reno*.

Ayudo a emprendedores a encontrar recursos en la región de Reno-Tahoe. ¡Hablo varios idiomas!

Cuéntame sobre ti y qué estás buscando. Por ejemplo:
• _"Soy veterano y quiero empezar un negocio"_
• _"Necesito financiamiento para mi startup"_
• _"¿Dónde puedo encontrar mentoría gratuita?"_

¿En qué puedo ayudarte hoy? 🚀""",

    "zh": """👋 你好 {name}！我是 *雷诺创业生态系统导航员*。

我帮助创业者在雷诺-太浩地区找到资源。我会说多种语言！

告诉我你的情况和需求。例如：
• _"我是退伍军人，想创业"_
• _"我需要为我的科技创业公司融资"_
• _"哪里可以找到免费的导师指导？"_

今天我能帮你什么？🚀""",

    "tl": """👋 Kumusta {name}! Ako ang *Reno Startup Ecosystem Navigator*.

Tinutulungan ko ang mga founder na makahanap ng resources sa Reno-Tahoe region. Nagsasalita ako ng maraming wika!

Sabihin mo sa akin ang tungkol sa iyo at kung ano ang hinahanap mo. Halimbawa:
• _"Veteran ako at gusto kong magsimula ng negosyo"_
• _"Kailangan ko ng funding para sa aking startup"_
• _"Saan ako makakahanap ng libreng mentorship?"_

Paano kita matutulungan ngayon? 🚀""",

    "vi": """👋 Xin chào {name}! Tôi là *Người hướng dẫn Hệ sinh thái Khởi nghiệp Reno*.

Tôi giúp các nhà sáng lập tìm kiếm nguồn lực ở khu vực Reno-Tahoe. Tôi nói được nhiều ngôn ngữ!

Hãy cho tôi biết về bạn và bạn đang tìm kiếm gì. Ví dụ:
• _"Tôi là cựu chiến binh muốn bắt đầu kinh doanh"_
• _"Tôi cần vốn cho startup công nghệ"_
• _"Tôi có thể tìm cố vấn miễn phí ở đâu?"_

Hôm nay tôi có thể giúp gì cho bạn? 🚀"""
}

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
# LANGUAGE DETECTION (simple heuristic for welcome message)
# ============================================================================

def detect_language_simple(text: str) -> str:
    """Simple language detection for initial greeting."""
    text_lower = text.lower()
    
    # Spanish indicators
    spanish_words = ['hola', 'necesito', 'ayuda', 'negocio', 'busco', 'quiero', 'soy', 'español']
    if any(word in text_lower for word in spanish_words):
        return "es"
    
    # Chinese characters
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return "zh"
    
    # Vietnamese indicators (with diacritics)
    vietnamese_chars = ['ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư', 'ạ', 'ả', 'ã', 'ầ', 'ẩ']
    if any(char in text_lower for char in vietnamese_chars):
        return "vi"
    
    # Tagalog indicators
    tagalog_words = ['ako', 'ang', 'mga', 'ko', 'sa', 'ng', 'kailangan', 'gusto', 'negosyo']
    if any(word in text_lower for word in tagalog_words):
        return "tl"
    
    # Korean characters
    if any('\uac00' <= char <= '\ud7af' for char in text):
        return "ko"
    
    # Japanese characters (hiragana/katakana)
    if any('\u3040' <= char <= '\u30ff' for char in text):
        return "ja"
    
    # Default to English
    return "en"

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
    """Determine if we should search and what query to use."""
    
    # Search indicators in multiple languages
    search_indicators = [
        # English
        "looking for", "need help", "find", "search", "recommend",
        "resources", "funding", "mentor", "accelerator", "workspace",
        "starting", "business", "startup", "entrepreneur", "founder",
        "veteran", "student", "woman", "grant", "investor",
        # Spanish
        "busco", "necesito", "encontrar", "ayuda", "negocio", "emprender",
        "financiamiento", "mentor", "subvención", "veterano",
        # Chinese
        "寻找", "需要", "帮助", "创业", "资金", "导师", "启动",
        # Vietnamese
        "tìm", "cần", "giúp", "kinh doanh", "khởi nghiệp", "vốn",
        # Tagalog
        "kailangan", "hanap", "tulong", "negosyo", "puhunan"
    ]
    
    message_lower = message.lower()
    
    # If it's clearly a search request
    if any(indicator in message_lower for indicator in search_indicators):
        return True, message
    
    # Short follow-ups in multiple languages
    short_responses = [
        "yes", "yeah", "sure", "ok", "okay", "tell me more", "more", "details",
        "sí", "si", "más", "detalles", "claro",
        "是", "好", "更多", "详细",
        "oo", "pa", "dagdag",
        "vâng", "có", "thêm"
    ]
    if message_lower.strip() in short_responses:
        return False, ""
    
    # For ambiguous cases, default to search
    return True, message

async def get_claude_response(user_id: int, user_message: str, search_results: str = None) -> str:
    """Get response from Claude with optional search results."""
    
    history = get_conversation_history(user_id)
    
    # Build the user message with search context if available
    if search_results and search_results != "NO RESULTS FOUND":
        augmented_message = f"""USER MESSAGE: {user_message}

SEARCH RESULTS FROM DATABASE:
{search_results}

IMPORTANT: Respond in the SAME LANGUAGE as the user's message. Analyze these results and provide a helpful, personalized response in their language."""
    elif search_results == "NO RESULTS FOUND":
        augmented_message = f"""USER MESSAGE: {user_message}

No resources were found matching this query. 
IMPORTANT: Respond in the SAME LANGUAGE as the user's message. Ask clarifying questions in their language to better understand what they need."""
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
        return "I'm having trouble connecting right now. Please try again in a moment. / Tengo problemas de conexión. Inténtalo de nuevo. / 连接出现问题，请稍后再试。"

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
    
    # Try to detect language from Telegram's language_code
    lang_code = user.language_code[:2] if user.language_code else "en"
    
    # Get welcome message in user's language, default to English
    welcome_text = WELCOME_MESSAGES.get(lang_code, WELCOME_MESSAGES["en"])
    welcome_text = welcome_text.format(name=user.first_name)
    
    await update.message.reply_text(welcome_text, parse_mode='Markdown')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    help_text = """*How to use this bot / Cómo usar / 如何使用:*

Just chat naturally in your language!
¡Solo chatea naturalmente en tu idioma!
用你的语言自然聊天！

*Commands / Comandos / 命令:*
/start - Start fresh / Empezar de nuevo / 重新开始
/clear - Clear history / Borrar historial / 清除历史
/lang - Change language / Cambiar idioma / 更改语言
/help - Show this message

🌐 Supported: English, Español, 中文, Tiếng Việt, Tagalog, 한국어, 日本語, and more!"""

    await update.message.reply_text(help_text, parse_mode='Markdown')

async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear conversation history."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    clear_history(update.effective_user.id)
    await update.message.reply_text(
        "🔄 Conversation cleared!\n"
        "¡Conversación borrada!\n"
        "对话已清除！\n\n"
        "What would you like to explore? / ¿Qué te gustaría explorar? / 你想探索什么？"
    )

async def lang_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show language options."""
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("⛔ Sorry, you're not authorized to use this bot.")
        return
    
    lang_text = """🌐 *Supported Languages / Idiomas / 语言*

Just write in your preferred language and I'll respond in the same language!

• 🇺🇸 English - Just type in English
• 🇪🇸 Español - Solo escribe en español  
• 🇨🇳 中文 - 用中文输入即可
• 🇻🇳 Tiếng Việt - Chỉ cần nhập tiếng Việt
• 🇵🇭 Tagalog - Mag-type lang sa Tagalog
• 🇰🇷 한국어 - 한국어로 입력하세요
• 🇯🇵 日本語 - 日本語で入力してください
• 🇫🇷 Français - Écrivez en français
• 🇩🇪 Deutsch - Schreiben Sie auf Deutsch
• 🇵🇹 Português - Escreva em português

Try it! / ¡Pruébalo! / 试试看！"""

    await update.message.reply_text(lang_text, parse_mode='Markdown')

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
            "Sorry, I encountered an error. Please try again or use /start to restart.\n"
            "Lo siento, ocurrió un error. Intenta de nuevo o usa /start.\n"
            "抱歉，出现错误。请重试或使用 /start。"
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
        
        # Get user's detected language from recent conversation
        history = get_conversation_history(query.from_user.id)
        user_lang = "en"
        if history:
            # Use last user message to detect language
            for msg in reversed(history):
                if msg["role"] == "user":
                    user_lang = detect_language_simple(msg["content"])
                    break
        
        try:
            resource = get_resource_details(resource_id)
            
            if resource:
                # Format detailed view
                cost = resource.get('cost_levels', {}).get('name', 'N/A') if resource.get('cost_levels') else 'N/A'
                cost_emoji = {"Free": "🆓", "Paid": "💰", "Varies": "💲"}.get(cost, "")
                
                # Basic info (keep resource names in English as proper nouns)
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
                    keyboard.append([InlineKeyboardButton("🌐 Website", url=resource['website'])])
                
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
    
    print("🤖 Starting Reno Ecosystem Telegram Bot (Multi-Language)...")
    print(f"🌐 Supported languages: {', '.join(SUPPORTED_LANGUAGES.values())}")
    
    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_command))
    application.add_handler(CommandHandler("lang", lang_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    application.add_error_handler(error_handler)
    
    # Start polling
    print("✅ Bot is running! Press Ctrl+C to stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
