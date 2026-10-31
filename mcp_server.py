import os
import yfinance as yf
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Initialize FastMCP Server
mcp = FastMCP("AlphaInsight-Financial-Tools", port=8000)

# Initialize ChromaDB Vector Store
persist_dir = "./chroma_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = None

if os.path.exists(persist_dir):
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)


@mcp.tool()
def get_live_market_data(ticker: str) -> str:
    """Fetch real-time financial market parameters from Yahoo Finance.
    Parameters:
    - ticker: Ticker symbol (e.g., 'NVDA', 'AAPL', 'MSFT').
    Returns:
    - Formatted string tagged with [LIVE MARKET DATA] including current price,
      52-week range, trailing P/E ratio, and market capitalization in native currency (USD).
    """
    clean_ticker = ticker.strip().upper().replace("$", "")
    print(f"[MCP-SERVER] Executing get_live_market_data for: {clean_ticker}")
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


@mcp.tool()
def search_annual_report(query: str, k: int = 3) -> str:
    """Search qualitative sections of 10-K filings and annual reports from persistent ChromaDB.
    Parameters:
    - query: Qualitative search terms (e.g., 'risk factors', 'revenue drivers', 'competition').
    - k: Number of chunks to retrieve (default is 3).
    Returns:
    - Formatted string tagged with [STATIC DOCUMENT DATA] citing report page numbers.
    """
    print(f"[MCP-SERVER] Executing search_annual_report for query: '{query}'")
    if not vectorstore:
        return "[STATIC DOCUMENT DATA] Error: Vector database not initialized on MCP server."

    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "[STATIC DOCUMENT DATA] No matching sections found in the report."

    extracted = []
    for doc in docs:
        page = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content.strip()
        extracted.append(f"[Page {page}]: {snippet}")

    return "[STATIC DOCUMENT DATA]\n" + "\n\n".join(extracted)


if __name__ == "__main__":
    # Runs the MCP server with SSE transport over dedicated port 8000
    mcp.run(transport="sse")