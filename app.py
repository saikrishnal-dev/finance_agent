import os
import re
import streamlit as st
import yfinance as yf

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MINIMAL DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AlphaInsight | Financial Agent",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional, high-legibility CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Minimal Top Ticker Cards */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: left;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.25rem;
        color: #f0f6fc;
        font-weight: 700;
    }
    .metric-sub {
        font-size: 0.75rem;
        color: #58a6ff;
        margin-top: 2px;
    }

    /* Source Badge Highlights for the Chat */
    .live-tag {
        display: inline-block;
        background-color: rgba(35, 134, 54, 0.2);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .static-tag {
        display: inline-block;
        background-color: rgba(31, 111, 235, 0.2);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.4);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    /* Disclaimer Notice */
    .disclaimer-banner {
        background-color: #0d1117;
        border-left: 3px solid #d29922;
        color: #8b949e;
        padding: 8px 12px;
        font-size: 0.75rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. VECTOR STORE & DATA RETRIEVERS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    persist_dir = "./chroma_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    return None

vectorstore = load_vectorstore()

@st.cache_data(ttl=60)
def fetch_live_summary(ticker: str):
    """Fetches clean parameters for the top metric cards."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "price": info.get("currentPrice") or info.get("regularMarketPrice", "N/A"),
            "currency": info.get("currency", "USD"),
            "pe": info.get("trailingPE") or info.get("forwardPE", "N/A"),
            "low": info.get("fiftyTwoWeekLow", "N/A"),
            "high": info.get("fiftyTwoWeekHigh", "N/A"),
            "cap": info.get("marketCap", "N/A")
        }
    except Exception:
        return None

def get_live_market_data(ticker: str) -> str:
    clean_ticker = ticker.strip().upper().replace("$", "")
    try:
        t = yf.Ticker(clean_ticker)
        info = t.info
        currency = info.get("currency", "USD")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice", "N/A")
        pe_ratio = info.get("trailingPE") or info.get("forwardPE", "N/A")
        high_52 = info.get("fiftyTwoWeekHigh", "N/A")
        low_52 = info.get("fiftyTwoWeekLow", "N/A")
        mkt_cap = info.get("marketCap", "N/A")
        
        return (
            f"[LIVE MARKET DATA]\n"
            f"- Ticker: {clean_ticker}\n"
            f"- Current Price: {current_price} {currency}\n"
            f"- 52-Week Range: {low_52} - {high_52} {currency}\n"
            f"- Trailing P/E: {pe_ratio}\n"
            f"- Market Cap: {mkt_cap} {currency}"
        )
    except Exception as e:
        return f"[LIVE MARKET DATA] Error fetching ticker {clean_ticker}: {str(e)}"

def search_annual_report(query: str, k_chunks: int = 3) -> str:
    if not vectorstore:
        return "[STATIC DOCUMENT DATA] Error: Vector database (chroma_db) not initialized."
    docs = vectorstore.similarity_search(query, k=k_chunks)
    if not docs:
        return "[STATIC DOCUMENT DATA] No relevant sections found in the report."
    
    extracted = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content.strip()
        extracted.append(f"[Page {page}]: {snippet}")
    return "[STATIC DOCUMENT DATA]\n" + "\n\n".join(extracted)

# -----------------------------------------------------------------------------
# 3. SIDEBAR CONTROLS & MODEL PARAMETERS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Agent Settings")
    selected_ticker = st.text_input("Active Ticker", value="NVDA").upper().strip()
    
    st.markdown("---")
    st.caption("Retrieval Parameters")
    retrieval_k = st.slider("Document Chunks ($k$)", min_value=1, max_value=6, value=3)
    model_temp = st.slider("Temperature", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
    
    st.markdown("---")
    st.caption("System Status")
    if vectorstore:
        st.success("● ChromaDB Connected", icon="✅")
    else:
        st.error("● ChromaDB Missing", icon="⚠️")
        
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 4. TOP STATS BAR (Precise, uncluttered parameters)
# -----------------------------------------------------------------------------
st.title("AlphaInsight")
st.markdown(
    '<div class="disclaimer-banner"><strong>Disclaimer:</strong> AI research prototype for educational/informational evaluation. Not certified financial advice.</div>',
    unsafe_allow_html=True
)

summary = fetch_live_summary(selected_ticker)
if summary:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">{selected_ticker} Live Price</div>
            <div class="metric-value">{summary['price']} <span style="font-size:0.8rem; font-weight:normal;">{summary['currency']}</span></div>
            <div class="metric-sub">Real-Time Quote</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">52-Week Range</div>
            <div class="metric-value" style="font-size:1rem; padding-top:4px;">{summary['low']} - {summary['high']}</div>
            <div class="metric-sub">{summary['currency']} High/Low</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        pe_val = f"{summary['pe']:.2f}" if isinstance(summary['pe'], (int, float)) else str(summary['pe'])
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Valuation (P/E)</div>
            <div class="metric-value">{pe_val}</div>
            <div class="metric-sub">Trailing Multiple</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Market Capitalization</div>
            <div class="metric-value" style="font-size:1.05rem;">{summary['cap']:,}</div>
            <div class="metric-sub">{summary['currency']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border: 0; border-top: 1px solid #21262d; margin: 20px 0;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. ORCHESTRATION & CHAT INTERFACE
# -----------------------------------------------------------------------------
llm = ChatOllama(model="llama3.2", temperature=model_temp)

def run_agent_workflow(user_query: str, ticker: str, k: int) -> str:
    q_lower = user_query.lower()
    needs_live = any(w in q_lower for w in ["price", "p/e", "pe", "valuation", "cap", "52-week", "range", "current", "trading"])
    needs_static = any(w in q_lower for w in ["report", "10-k", "risk", "filing", "revenue", "annual", "segment", "driver", "strategy"])

    # Default to hybrid if query is broad
    if not needs_live and not needs_static:
        needs_live, needs_static = True, True

    context_parts = []
    if needs_live:
        context_parts.append(get_live_market_data(ticker))
    if needs_static:
        search_query = "key risk factors business drivers revenue" if "risk" in q_lower or "driver" in q_lower else user_query
        context_parts.append(search_annual_report(search_query, k_chunks=k))

    context_str = "\n\n".join(context_parts)

    synthesis_prompt = (
        "You are AlphaInsight, an autonomous investment research agent.\n"
        "Synthesize the verified data provided below into a clean, concise, non-redundant briefing.\n\n"
        "STRICT SOURCE ATTRIBUTION RULES:\n"
        "1. [LIVE MARKET DATA] must contain ONLY real-time figures from Yahoo Finance (stock price, 52-week range, P/E, market cap).\n"
        "2. [STATIC DOCUMENT DATA] must contain ONLY qualitative risks, business drivers, or financial footnotes from the 10-K/annual report.\n"
        "   - Do NOT include live valuation metrics (P/E or price) inside the document section.\n"
        "3. Explicitly state all units and currencies.\n"
        "4. Conclude with a neutral, objective financial summary without buy/sell recommendations.\n\n"
        f"DATA SOURCES:\n{context_str}\n\n"
        f"QUESTION: {user_query}"
    )

    response = llm.invoke([
        SystemMessage(content="You are a precise financial research assistant."),
        HumanMessage(content=synthesis_prompt)
    ])
    return response.content

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ready. Ask about live valuation parameters or qualitative filing risks."}
    ]

# Render conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_prompt := st.chat_input("Enter financial query (e.g., 'Compare current price and P/E against core report risks')..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving verified parameters..."):
            answer = run_agent_workflow(user_prompt, selected_ticker, retrieval_k)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})