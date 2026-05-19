import asyncio

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from rag.logger import setup_logging

console = Console()


@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug-level logs")
def main(debug: bool) -> None:
    """Interactive chat with your indexed codebase."""
    import logging
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    console.print(Panel("[bold cyan]Modular RAG — Codebase Assistant[/bold cyan]\nType [yellow]exit[/yellow] to quit."))
    asyncio.run(_chat_loop())


async def _chat_loop() -> None:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    while True:
        question = Prompt.ask("\n[bold green]You[/bold green]").strip()
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Bye.[/dim]")
            break
        if not question:
            continue

        console.print("\n[bold blue]Assistant[/bold blue]")
        answer_parts: list[str] = []
        citations = []

        stream = await pipeline.query(question)
        async for chunk in stream:
            console.print(chunk.text, end="", highlight=False)
            answer_parts.append(chunk.text)
            citations = chunk.citations  # last chunk has all citations

        console.print()  # newline after streamed answer

        if citations:
            console.print("\n[dim]Sources:[/dim]")
            seen: set[str] = set()
            for c in citations:
                key = f"{c.file_path}:{c.start_line}"
                if key not in seen:
                    seen.add(key)
                    console.print(f"  [dim cyan]{c.file_path}[/dim cyan] [dim]lines {c.start_line}–{c.end_line}[/dim]")


if __name__ == "__main__":
    main()
