"""
RAG (Retrieval-Augmented Generation) module for AI Assistant.

Uses OpenAI embeddings to create a vector store from session data
and retrieves relevant context for answering questions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import numpy as np


@dataclass
class Document:
    """A document chunk with metadata"""
    content: str
    metadata: dict[str, Any]
    embedding: Optional[np.ndarray] = None


class VectorStore:
    """Simple in-memory vector store with cosine similarity search"""

    def __init__(self):
        self.documents: list[Document] = []

    def add_documents(self, documents: list[Document]):
        """Add documents to the store"""
        self.documents.extend(documents)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[Document]:
        """Find top-k most similar documents using cosine similarity"""
        if not self.documents or query_embedding is None:
            return []

        # Calculate cosine similarity for all documents
        similarities = []
        for doc in self.documents:
            if doc.embedding is None:
                continue
            # Cosine similarity = dot product of normalized vectors
            sim = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding) + 1e-10
            )
            similarities.append((doc, float(sim)))

        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in similarities[:k]]

    def clear(self):
        """Clear all documents from the store"""
        self.documents.clear()


class OpenAIEmbedder:
    """OpenAI embeddings client"""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.openai.com/v1"

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Get embeddings for a list of texts"""
        if not texts:
            return []

        # OpenAI API accepts up to 2048 texts per request for embeddings
        # We'll process in batches of 100 to be safe
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            payload = {
                "model": self.model,
                "input": batch,
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    res = await client.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=payload
                    )
                    res.raise_for_status()
                    data = res.json()

                    # Extract embeddings in order
                    embeddings_data = data.get("data", [])
                    embeddings_data.sort(key=lambda x: x.get("index", 0))

                    for item in embeddings_data:
                        embedding = item.get("embedding", [])
                        all_embeddings.append(np.array(embedding, dtype=np.float32))

                except Exception as e:
                    print(f"Error getting embeddings: {e}")
                    # Return zero vectors as fallback
                    all_embeddings.extend([np.zeros(1536, dtype=np.float32) for _ in batch])

        return all_embeddings


class RAGRetriever:
    """Main RAG retriever class"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.embedder = OpenAIEmbedder(self.api_key) if self.api_key else None
        self.vector_store = VectorStore()
        self.initialized = False

    def chunk_data(self, data: dict[str, Any]) -> list[Document]:
        """Convert data packet into semantic chunks"""
        documents = []

        # Chunk 1: Raw sales summary
        try:
            if data.get("raw_sales"):
                rs = data["raw_sales"]
                content = f"""Raw Sales Data Summary:
                    - Date Range: {rs.get('date_range', {}).get('min', 'N/A')} to {rs.get('date_range', {}).get('max', 'N/A')}
- Total Rows: {rs.get('metrics', {}).get('rows', 0)}
- Total Sales: {rs.get('metrics', {}).get('total_sales', 0):.2f}
- Average Sales: {rs.get('metrics', {}).get('avg_sales', 0):.2f}
- Max Sales: {rs.get('metrics', {}).get('max_sales', 0):.2f}
- Available Columns: {', '.join(rs.get('available_columns', []))}"""

                documents.append(Document(
                    content=content,
                    metadata={"type": "raw_sales_summary"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 1 (raw_sales_summary) failed: {e}")

        # Chunk 2: Monthly raw sales rollup
        try:
            if data.get("raw_sales", {}).get("monthly_rollup"):
                monthly = data["raw_sales"]["monthly_rollup"]
                if monthly:
                    content = "Monthly Raw Sales:\n"
                    for entry in monthly[-12:]:  # Last 12 months
                        month = entry.get("month", "")
                        sales = entry.get("total_sales", 0)
                        content += f"- {month}: {sales:.2f} units\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "raw_sales_monthly"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 2 (raw_sales_monthly) failed: {e}")

        # Chunk 3: Forecast summary
        try:
            if data.get("forecast_output"):
                fc = data["forecast_output"]
                content = f"""Forecast Summary:
                    - Date Range: {fc.get('date_range', {}).get('min', 'N/A')} to {fc.get('date_range', {}).get('max', 'N/A')}
- Total Rows: {fc.get('metrics', {}).get('rows', 0)}
- Total Forecast: {fc.get('metrics', {}).get('total_forecast', 0):.2f}
- Total Actual: {fc.get('metrics', {}).get('total_actual', 0):.2f}
- Forecast Column: {fc.get('metrics', {}).get('forecast_col', 'N/A')}
- Available Columns: {', '.join(fc.get('available_columns', []))}"""

                documents.append(Document(
                    content=content,
                    metadata={"type": "forecast_summary"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 3 (forecast_summary) failed: {e}")

        # Chunk 4: Monthly forecast rollup
        try:
            if data.get("forecast_output", {}).get("monthly_rollup"):
                monthly = data["forecast_output"]["monthly_rollup"]
                if monthly:
                    content = "Monthly Forecast:\n"
                    for entry in monthly[-12:]:
                        month = entry.get("month", "")
                        forecast = entry.get("forecast_units", 0)
                        actual = entry.get("actual_units", 0)
                        content += f"- {month}: Forecast={forecast:.2f}, Actual={actual:.2f}\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "forecast_monthly"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 4 (forecast_monthly) failed: {e}")

        # Chunk 5: Supply plan summary
        try:
            if data.get("supply_plan"):
                sp = data["supply_plan"]
                content = f"""Supply Plan Summary:
- Total Rows: {sp.get('metrics', {}).get('rows', 0)}
- Total Stockout Units: {sp.get('metrics', {}).get('total_stockout_units', 0):.2f}
- Months with Stockout: {sp.get('metrics', {}).get('months_with_stockout', 0)}
- Total Order Quantity: {sp.get('metrics', {}).get('total_order_qty', 0):.2f}
- Available Columns: {', '.join(sp.get('available_columns', []))}"""

                documents.append(Document(
                    content=content,
                    metadata={"type": "supply_plan_summary"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 5 (supply_plan_summary) failed: {e}")

        # Chunk 6: Supply plan details (sample rows)
        try:
            if data.get("supply_plan", {}).get("sample_rows"):
                rows = data["supply_plan"]["sample_rows"]
                if rows:
                    content = "Supply Plan Details (recent periods):\n"
                    for row in rows[-10:]:  # Last 10 rows
                        content += f"- Period: {row.get('period_start', 'N/A')}, "
                        content += f"Demand: {row.get('forecast_demand', 0):.0f}, "
                        content += f"Order Qty: {row.get('order_qty', 0):.0f}, "
                        content += f"Ending Inventory: {row.get('ending_on_hand', 0):.0f}, "
                        content += f"Risk: {row.get('risk_flag', 'N/A')}\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "supply_plan_details"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 6 (supply_plan_details) failed: {e}")

        # Chunk 7: Numeric answers (authoritative calculations)
        try:
            if data.get("numeric_answers"):
                na = data["numeric_answers"]
                content = "Authoritative Numeric Calculations:\n"

                if na.get("dec_2024_total_sales") is not None:
                    content += f"- December 2024 Total Sales: {na['dec_2024_total_sales']:.2f}\n"

                if na.get("raw_monthly_rollup"):
                    content += "\nRaw Sales Monthly Breakdown:\n"
                    for entry in na["raw_monthly_rollup"][-6:]:  # Last 6 months
                        content += f"- {entry.get('month', 'N/A')}: {entry.get('total', 0):.2f}\n"

                if na.get("forecast_monthly_rollup"):
                    content += "\nForecast Monthly Breakdown:\n"
                    for entry in na["forecast_monthly_rollup"][-6:]:
                        content += f"- {entry.get('month', 'N/A')}: {entry.get('total', 0):.2f}\n"

                documents.append(Document(
                    content=content,
                    metadata={"type": "numeric_answers"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 8 (numeric_answers) failed: {e}")

        # Chunk 8: Grain/filter information
        try:
            grain = data.get("grain", {})
            if grain.get("sku") or grain.get("location"):
                content = f"""Current Filter/Grain:
- SKU/Item: {grain.get('sku', 'All')}
- Location/Store: {grain.get('location', 'All')}
- Combo Key: {grain.get('combo_key', 'N/A')}"""

                documents.append(Document(
                    content=content,
                    metadata={"type": "grain_info"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 8 (grain_info) failed: {e}")

        # Chunk 9: Forecast drivers and model features
        # This helps answer questions about what drives the forecast
        try:
            if data.get("forecast_output"):
                fc = data["forecast_output"]
                available_cols = fc.get("available_columns", [])

                # Identify feature columns (exclude date, forecast columns, and actual)
                feature_cols = [col for col in available_cols if not any(x in col.lower() for x in
                    ['date', 'forecast', 'actual', 'p10', 'p50', 'p90'])]

                content = """Forecast Model Information:

The forecast is generated using a quantile regression model (LightGBM) that predicts multiple quantiles (P10, P50, P90) simultaneously.

Key Features Used in Forecasting:
1. **Temporal Features**: The model uses date-based features to capture time trends and patterns
2. **Seasonal Features**: Built-in seasonality indicators like Q4 flag, months to December, pre-peak periods, and peak build periods
3. **Historical Patterns**: The model learns from historical sales data to identify recurring patterns
"""

                if feature_cols:
                    content += f"4. **Additional Features**: {', '.join(feature_cols[:10])} - these categorical or numeric features from your data help the model understand different segments and patterns\n"

                # Add insights based on monthly rollup if available
                if data.get("forecast_output", {}).get("monthly_rollup"):
                    monthly = data["forecast_output"]["monthly_rollup"]
                    if monthly and isinstance(monthly, list) and len(monthly) >= 6:
                        # Analyze seasonality in last 6 months
                        recent = [m for m in monthly[-6:] if m is not None]
                        if recent:
                            avg_forecast = sum(m.get("forecast_units", 0) for m in recent if isinstance(m, dict)) / len(recent)
                            valid_months = [m for m in recent if isinstance(m, dict) and m.get("forecast_units") is not None]
                            if valid_months:
                                max_month = max(valid_months, key=lambda x: x.get("forecast_units", 0))

                                content += f"\n**Seasonal Patterns Detected**:\n"
                                content += f"- Recent average monthly forecast: {avg_forecast:.0f} units\n"
                                content += f"- Peak month in recent period: {max_month.get('month', 'N/A')} with {max_month.get('forecast_units', 0):.0f} units\n"
                                content += f"- The model has learned these seasonal patterns from historical data and applies them to future predictions\n"

                content += """\n**Main Drivers of Forecasts**:
1. Historical sales trends and volumes
2. Seasonal patterns (Q4 peaks, monthly cycles)
3. Time-based trends (overall growth or decline)
"""

                if feature_cols:
                    content += f"4. Segmentation by {', '.join(feature_cols[:3])} which allows different patterns for different segments\n"

                documents.append(Document(
                    content=content,
                    metadata={"type": "model_features_and_drivers"}
                ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 9 (model_features_and_drivers) failed: {e}")

        # Chunk 10: Historical patterns and trends
        try:
            if data.get("raw_sales", {}).get("monthly_rollup"):
                monthly = data["raw_sales"]["monthly_rollup"]
                if monthly and isinstance(monthly, list) and len(monthly) >= 12:
                    content = "Historical Sales Patterns and Trends:\n\n"

                    # Filter out None values
                    valid_monthly = [m for m in monthly if m is not None and isinstance(m, dict)]
                    if len(valid_monthly) >= 12:
                        # Calculate year-over-year growth if we have enough data
                        recent_12 = valid_monthly[-12:]
                        total_recent = sum(m.get("total_sales", 0) for m in recent_12)

                        if len(valid_monthly) >= 24:
                            prior_12 = valid_monthly[-24:-12]
                            total_prior = sum(m.get("total_sales", 0) for m in prior_12)
                            if total_prior > 0:
                                yoy_growth = ((total_recent - total_prior) / total_prior) * 100
                                content += f"- Year-over-Year Growth: {yoy_growth:.1f}%\n"

                        # Identify seasonal patterns
                        q4_months = [m for m in recent_12 if isinstance(m, dict) and m.get("month", "")[-2:] in ["10", "11", "12"]]
                        if q4_months:
                            q4_total = sum(m.get("total_sales", 0) for m in q4_months)
                            q4_pct = (q4_total / total_recent * 100) if total_recent > 0 else 0
                            content += f"- Q4 Seasonality: Q4 months represent {q4_pct:.1f}% of annual sales, showing {'strong' if q4_pct > 30 else 'moderate'} seasonal impact\n"

                        # Identify peak month
                        if recent_12:
                            peak_month = max(recent_12, key=lambda x: x.get("total_sales", 0) if isinstance(x, dict) else 0)
                            if isinstance(peak_month, dict):
                                content += f"- Peak Sales Month: {peak_month.get('month', 'N/A')} with {peak_month.get('total_sales', 0):.0f} units\n"

                        content += "\nThese historical patterns are the foundation that the forecasting model uses to predict future demand."

                        documents.append(Document(
                            content=content,
                            metadata={"type": "historical_patterns"}
                        ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 10 (historical_patterns) failed: {e}")

        # Chunk 11: Raw sales by item (comprehensive item-level data)
        try:
            if data.get("numeric_answers", {}).get("raw_sales_by_item"):
                items = data["numeric_answers"]["raw_sales_by_item"]
                if items and isinstance(items, list):
                    content = "Complete Item-Level Sales Data (All Items):\n\n"
                    content += "This data includes ALL items with their total historical sales. Use this to answer questions about specific items, item rankings, or item comparisons.\n\n"

                    for item in items:
                        if isinstance(item, dict):
                            item_id = item.get("item", "Unknown")
                            total = item.get("total_sales", 0)
                            content += f"- Item {item_id}: {total:.2f} total sales\n"

                    total_items = len(items)
                    content += f"\nTotal unique items: {total_items}\n"
                    content += "\nYou can use this data to:\n"
                    content += "- Find the item with highest/lowest sales\n"
                    content += "- Compare sales between specific items\n"
                    content += "- Calculate average sales per item\n"
                    content += "- Identify items above/below certain thresholds\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "raw_sales_by_item"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 11 (raw_sales_by_item) failed: {e}")

        # Chunk 12: Raw sales by store (comprehensive store-level data)
        try:
            if data.get("numeric_answers", {}).get("raw_sales_by_store"):
                stores = data["numeric_answers"]["raw_sales_by_store"]
                if stores and isinstance(stores, list):
                    content = "Complete Store-Level Sales Data (All Stores):\n\n"
                    content += "This data includes ALL stores/locations with their total historical sales. Use this to answer questions about specific stores, store rankings, or store comparisons.\n\n"

                    for store in stores:
                        if isinstance(store, dict):
                            store_id = store.get("location", "Unknown")
                            total = store.get("total_sales", 0)
                            content += f"- Store {store_id}: {total:.2f} total sales\n"

                    total_stores = len(stores)
                    content += f"\nTotal unique stores: {total_stores}\n"
                    content += "\nYou can use this data to:\n"
                    content += "- Find the store with highest/lowest sales\n"
                    content += "- Compare sales between specific stores\n"
                    content += "- Calculate average sales per store\n"
                    content += "- Identify stores above/below certain thresholds\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "raw_sales_by_store"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 12 (raw_sales_by_store) failed: {e}")

        # Chunk 13: Forecast by item (comprehensive item-level forecast data)
        try:
            if data.get("numeric_answers", {}).get("forecast_by_item"):
                items = data["numeric_answers"]["forecast_by_item"]
                if items and isinstance(items, list):
                    content = "Complete Item-Level Forecast Data (All Items):\n\n"
                    content += "This data includes ALL items with their total forecast values. Use this to answer questions about forecasted demand by item.\n\n"

                    for item in items:
                        if isinstance(item, dict):
                            item_id = item.get("item", "Unknown")
                            total = item.get("total_forecast", 0)
                            content += f"- Item {item_id}: {total:.2f} total forecasted units\n"

                    total_items = len(items)
                    content += f"\nTotal unique items: {total_items}\n"
                    content += "\nYou can use this data to:\n"
                    content += "- Find the item with highest/lowest forecast\n"
                    content += "- Compare forecasts between specific items\n"
                    content += "- Calculate total forecast across all/subset of items\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "forecast_by_item"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 13 (forecast_by_item) failed: {e}")

        # Chunk 14: Forecast by store (comprehensive store-level forecast data)
        try:
            if data.get("numeric_answers", {}).get("forecast_by_store"):
                stores = data["numeric_answers"]["forecast_by_store"]
                if stores and isinstance(stores, list):
                    content = "Complete Store-Level Forecast Data (All Stores):\n\n"
                    content += "This data includes ALL stores/locations with their total forecast values. Use this to answer questions about forecasted demand by store.\n\n"

                    for store in stores:
                        if isinstance(store, dict):
                            store_id = store.get("location", "Unknown")
                            total = store.get("total_forecast", 0)
                            content += f"- Store {store_id}: {total:.2f} total forecasted units\n"

                    total_stores = len(stores)
                    content += f"\nTotal unique stores: {total_stores}\n"
                    content += "\nYou can use this data to:\n"
                    content += "- Find the store with highest/lowest forecast\n"
                    content += "- Compare forecasts between specific stores\n"
                    content += "- Calculate total forecast across all/subset of stores\n"

                    documents.append(Document(
                        content=content,
                        metadata={"type": "forecast_by_store"}
                    ))
        except Exception as e:
            print(f"[RAG CHUNK ERROR] Chunk 14 (forecast_by_store) failed: {e}")

        return documents

    async def index_data(self, data: dict[str, Any]):
        """Index data packet into vector store"""
        if not self.embedder:
            return

        # Clear existing documents
        self.vector_store.clear()

        # Create document chunks
        documents = self.chunk_data(data)

        if not documents:
            return

        # Get embeddings for all documents
        texts = [doc.content for doc in documents]
        embeddings = await self.embedder.embed(texts)

        # Attach embeddings to documents
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        # Add to vector store
        self.vector_store.add_documents(documents)
        self.initialized = True

    async def retrieve(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k relevant documents for a query"""
        if not self.embedder or not self.initialized:
            return []

        # Get query embedding
        query_embeddings = await self.embedder.embed([query])
        if not query_embeddings:
            return []

        query_embedding = query_embeddings[0]

        # Search vector store
        results = self.vector_store.search(query_embedding, k=k)

        # Convert to serializable format
        return [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "relevance": "high"  # We already filtered by similarity
            }
            for doc in results
        ]


# Global retriever instance (initialized on first use)
_global_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get or create global retriever instance"""
    global _global_retriever
    if _global_retriever is None:
        _global_retriever = RAGRetriever()
    return _global_retriever


async def index_session_data(data: dict[str, Any]):
    """Index session data for retrieval"""
    retriever = get_retriever()
    await retriever.index_data(data)


async def retrieve(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Retrieve relevant context for a query"""
    retriever = get_retriever()
    return await retriever.retrieve(query, k=k)
