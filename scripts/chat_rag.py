#!/usr/bin/env python3
"""
Interactive chat interface for RAG queries.

Allows multiple questions in a session with conversation history.
Uses Turbopuffer for vector search and OpenAI for answer generation.

Example:
    python scripts/chat_rag.py
    python scripts/chat_rag.py --company 0000320193  # Apple
    python scripts/chat_rag.py --top-k 10
"""

import argparse
import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from turbopuffer import Turbopuffer

from public_company_rag.config import (
    get_turbopuffer_api_key,
    get_turbopuffer_region,
    get_turbopuffer_namespace,
    get_settings,
)
from public_company_rag.query import (
    semantic_search,
    query_by_company,
    generate_answer,
)


class RAGChat:
    """Interactive RAG chat session with conversation history."""

    def __init__(self, client: Turbopuffer, namespace_name: str, company_cik: str | None = None, top_k: int = 10):
        self.client = client
        self.namespace_name = namespace_name
        self.company_cik = company_cik
        self.top_k = top_k
        self.history: list[dict[str, str]] = []

    def clear_history(self):
        """Clear conversation history."""
        self.history = []

    def ask(self, question: str) -> dict:
        """
        Ask a question and generate an answer with conversation history.

        Returns:
            dict with 'answer', 'chunks', 'companies_mentioned', and 'timing'
        """
        # Retrieve relevant chunks (returns chunks and timing)
        if self.company_cik:
            chunks, search_timing = query_by_company(
                self.client,
                self.namespace_name,
                self.company_cik,
                question,
                top_k=self.top_k,
            )
        else:
            chunks, search_timing = semantic_search(
                self.client,
                self.namespace_name,
                question,
                top_k=self.top_k,
            )

        # Generate answer with conversation history (returns answer and timing)
        answer, generation_time = self._generate_answer_with_history(question, chunks)

        # Extract companies mentioned
        companies_mentioned = list({
            f"{chunk['company_ticker']}" if chunk['company_ticker'] else chunk['company_name']
            for chunk in chunks
            if chunk['company_name']
        })

        # Add to history
        self.history.append({
            "question": question,
            "answer": answer,
        })

        # Combine timing info
        timing = {
            "embedding": search_timing["embedding"],
            "search": search_timing["search"],
            "generation": generation_time,
            "total": search_timing["embedding"] + search_timing["search"] + generation_time,
        }

        return {
            "answer": answer,
            "chunks": chunks,
            "companies_mentioned": companies_mentioned,
            "timing": timing,
        }

    def _generate_answer_with_history(self, question: str, chunks: list[dict]) -> tuple[str, float]:
        """Generate answer considering conversation history."""
        import time
        from openai import OpenAI
        from public_company_rag.config import get_settings

        settings = get_settings()
        client = OpenAI(api_key=settings.openai_api_key)

        if not chunks:
            return "No relevant information found to answer the question.", 0.0

        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            company_info = f"{chunk['company_name']} ({chunk['company_ticker']})" if chunk['company_ticker'] else chunk['company_name']
            section_info = f"{chunk['section_type']} - {chunk['filing_year']}" if chunk['filing_year'] else chunk['section_type']

            context_parts.append(
                f"[{i}] {company_info} - {section_info}\n{chunk['text']}\n"
            )

        context_text = "\n".join(context_parts)

        # Build messages with conversation history
        system_prompt = """You are a financial analyst assistant. Answer questions based on the provided context from 10-K filings.
Be accurate and concise. If the context doesn't contain enough information to answer fully, acknowledge this.
If there's conversation history, use it to provide context-aware answers.
Cite sources using [N] notation where appropriate."""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (last 3 exchanges to keep context manageable)
        for hist in self.history[-3:]:
            messages.append({"role": "user", "content": hist["question"]})
            messages.append({"role": "assistant", "content": hist["answer"]})

        # Add current question with context
        user_prompt = f"""Context from 10-K filings:

{context_text}

Question: {question}

Please provide a clear answer based on the context above. Include citations using [N] notation where appropriate."""

        messages.append({"role": "user", "content": user_prompt})

        # Generate answer (timed)
        generation_start = time.time()
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.3,
        )
        generation_time = time.time() - generation_start

        return response.choices[0].message.content, generation_time


def main():
    parser = argparse.ArgumentParser(description="Interactive chat interface for RAG")
    parser.add_argument(
        "--company",
        type=str,
        help="Focus on a specific company CIK for all queries (e.g., 0000320193 for Apple)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve per query (default: 10)",
    )

    args = parser.parse_args()

    # Validate settings
    try:
        settings = get_settings()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please check your .env file")
        return 1

    # Connect to Turbopuffer
    print("Connecting to Turbopuffer...")
    try:
        client = Turbopuffer(
            api_key=get_turbopuffer_api_key(),
            region=get_turbopuffer_region(),
        )
        namespace_name = get_turbopuffer_namespace()

        # Verify namespace exists
        namespace = client.namespace(namespace_name)
        if not namespace.exists():
            print(f"Error: Namespace '{namespace_name}' does not exist")
            print("Please run setup first: python scripts/setup_data.py")
            return 1

        print("✓ Connected successfully\n")
    except Exception as e:
        print(f"Failed to connect to Turbopuffer: {e}")
        return 1

    # Create chat session
    chat = RAGChat(
        client=client,
        namespace_name=namespace_name,
        company_cik=args.company,
        top_k=args.top_k,
    )

    print("=" * 80)
    print("RAG Chat Interface")
    print("=" * 80)
    print()
    print(f"Top-K: {args.top_k}")
    if args.company:
        print(f"Focus: Company CIK {args.company}")
    print()
    print("Type your questions below. Commands:")
    print("  /quit or /exit - Exit")
    print("  /clear - Clear conversation history")
    print("  /help - Show this help")
    print()
    print("-" * 80)
    print()

    try:
        while True:
            try:
                question = input("Question: ").strip()

                if not question:
                    continue

                # Handle commands
                if question.lower() in ["/quit", "/exit", "quit", "exit"]:
                    print("\nGoodbye!")
                    break

                if question.lower() in ["/clear", "clear"]:
                    chat.clear_history()
                    print("Conversation history cleared\n")
                    continue

                if question.lower() in ["/help", "help"]:
                    print("\nCommands:")
                    print("  /quit or /exit - Exit")
                    print("  /clear - Clear conversation history")
                    print("  /help - Show this help")
                    print()
                    continue

                print()

                # Ask question
                response = chat.ask(question)

                # Display timing breakdown (compact, one line)
                timing = response['timing']
                print(f"Timing: {timing['embedding']:.2f}s embed + {timing['search']:.2f}s search + {timing['generation']:.2f}s generation = {timing['total']:.2f}s total")

                # Display company summary
                num_companies = len(response['companies_mentioned'])
                if num_companies > 0:
                    company_list = ', '.join(response['companies_mentioned'])
                    print(f"Found {num_companies} companies in {len(response['chunks'])} chunks: {company_list}")
                else:
                    print(f"Searched {len(response['chunks'])} chunks")
                print()

                print("=" * 80)
                print("ANSWER:")
                print("=" * 80)
                print()
                print(response['answer'])
                print()
                print("=" * 80)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
                print()

    finally:
        pass  # Turbopuffer client doesn't need explicit close

    return 0


if __name__ == "__main__":
    sys.exit(main())
