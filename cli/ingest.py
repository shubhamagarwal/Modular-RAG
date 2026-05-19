import asyncio

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag.logger import setup_logging

console = Console()


@click.command()
@click.argument("path", default=".", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, default=False, help="Enable debug-level logs")
def main(path: str, debug: bool) -> None:
    """Index a codebase directory into ChromaDB."""
    import logging
    setup_logging(level=logging.DEBUG if debug else logging.INFO)
    asyncio.run(_ingest(path))


async def _ingest(path: str) -> None:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"Indexing [cyan]{path}[/cyan]…", total=None)
        result = await pipeline.ingest(path)
        progress.remove_task(task)

    console.print(f"[green]Done.[/green] {result.files_processed} files → {result.chunks_added} chunks in {result.duration_s}s")


if __name__ == "__main__":
    main()
