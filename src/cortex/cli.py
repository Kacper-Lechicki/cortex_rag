import typer
import warnings

from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table

app = typer.Typer(
    help="Cortex - your personal technical knowledge worker.",
    no_args_is_help=False,
    rich_markup_mode="rich",
)

console = Console()

# Some heavy deps (e.g. Chroma/ML runtimes) can trigger Python's multiprocessing
# `resource_tracker` warning about leaked semaphores at interpreter shutdown.
# This is noisy but typically harmless for CLI users; filter it narrowly.
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be \d+ leaked semaphore objects to clean up at shutdown",
    category=UserWarning,
    module=r"multiprocessing\.resource_tracker",
)


def _format_retrieval_context(results: list) -> str:
    """
    Format retrieved chunks into a citation-friendly context block.

    We include per-chunk URL + title so the generator can ground and cite.
    """

    blocks: list[str] = []

    # Keep individual chunks reasonably small to avoid blowing up the prompt.
    # (The generator also enforces its own overall cap.)
    max_chunk_chars = 1_800

    for i, r in enumerate(results, start=1):
        md = getattr(r, "metadata", {}) or {}
        title = (md.get("title") or "").strip()
        url = (md.get("source") or getattr(r, "source", "") or "").strip()
        domain = (md.get("domain") or "").strip()

        header_parts = [f"[Chunk {i}]"]
        if title:
            header_parts.append(f"Title: {title}")
        if domain:
            header_parts.append(f"Domain: {domain}")
        if url:
            header_parts.append(f"URL: {url}")

        content = (getattr(r, "content", "") or "").strip()
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars].rstrip() + "…"

        blocks.append("\n".join([" | ".join(header_parts), content]))

    return "\n\n---\n\n".join(blocks).strip()


def _ensure_questions_file(path: Path) -> bool:
    """
    Ensure evaluation questions JSON exists and is valid.

    Returns:
        True if file exists and parses as JSON list, or was created successfully.
        False if the user chose not to create/fix it.
    """

    import json

    from json import JSONDecodeError

    default_template = [
        {
            "id": "q1",
            "question": "What is this knowledge base about?",
            "relevant_source": "https://example.com/article",
        },
        {
            "id": "q2",
            "question": "Summarize the key points from the ingested content.",
            "relevant_source": "https://example.com/article",
        },
    ]

    if not path.exists():
        console.print(f"[yellow]Questions file not found:[/yellow] {path}")

        if not typer.confirm("Create a template questions file now?", default=True):
            return False

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(default_template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        console.print(f"[green]Created:[/green] {path}")

        return True

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else None

        if not isinstance(data, list):
            raise ValueError("JSON must be a list of question objects.")

        return True
    except (JSONDecodeError, ValueError) as e:
        console.print(f"[yellow]Questions file is invalid:[/yellow] {path}")
        console.print(f"[dim]{e}[/dim]")

        if not typer.confirm("Overwrite with a valid template?", default=True):
            return False

        path.write_text(
            json.dumps(default_template, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        console.print(f"[green]Overwritten with template:[/green] {path}")

        return True


def _load_questions(path: Path) -> list[dict] | None:
    """
    Load questions JSON as a list of dicts.

    Returns:
        - [] if file does not exist
        - list[dict] if valid
        - None if file exists but is invalid
    """

    import json

    from json import JSONDecodeError

    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw) if raw.strip() else None

        if not isinstance(data, list):
            return None

        # Ensure elements are dict-like; ignore non-dicts.
        return [x for x in data if isinstance(x, dict)]
    except (OSError, JSONDecodeError):
        return None


def _source_has_questions(questions: list[dict], source_url: str) -> bool:
    return any(q.get("relevant_source") == source_url for q in questions)


def _sources_missing_questions(
    sources: list[dict], questions: list[dict]
) -> list[dict]:
    return [s for s in sources if not _source_has_questions(questions, s["source"])]


def _write_questions_atomic(path: Path, questions: list[dict]) -> None:
    import json
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(questions, ensure_ascii=False, indent=2) + "\n"

    fd, tmp_path = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)

        os.replace(tmp_path, path)
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _create_questions_for_source(path: Path, source_url: str) -> bool:
    """
    Ensure `path` contains 2 template questions for `source_url` if none exist.

    Returns:
        True if questions exist after the call, False if user aborted.
    """

    import hashlib

    questions = _load_questions(path)

    if questions is None:
        console.print(f"[yellow]Questions file is invalid:[/yellow] {path}")

        if not typer.confirm(
            "Overwrite with a valid template for this source?", default=True
        ):
            return False

        questions = []

    if _source_has_questions(questions, source_url):
        console.print("[dim]Questions already exist for this source.[/dim]")
        return True

    h = hashlib.sha256(source_url.encode()).hexdigest()[:10]

    q1_id = f"{h}-1"
    q2_id = f"{h}-2"

    questions.extend(
        [
            {
                "id": q1_id,
                "question": "Summarize the key points from this source.",
                "relevant_source": source_url,
            },
            {
                "id": q2_id,
                "question": "What are 3 important facts or takeaways from this source?",
                "relevant_source": source_url,
            },
        ]
    )

    _write_questions_atomic(path, questions)

    console.print(f"[green]Questions created for:[/green] {source_url}")

    return True


def _prompt_menu_choice(prompt: str, valid: set[str]) -> str:
    while True:
        choice = console.input(prompt).strip().lower()

        if choice in valid:
            return choice

        console.print("[red]Invalid choice.[/red]")


def _run_menu() -> None:
    """
    Interactive menu UI (default when running `cortex` without a subcommand).
    """

    while True:
        console.print(
            Panel(
                "[bold cyan]Cortex[/bold cyan]\n"
                "[dim]Choose an option (or 0 to exit).[/dim]",
                border_style="cyan",
            )
        )

        table = Table(show_header=True, header_style="bold cyan")

        table.add_column("Category", style="bold")
        table.add_column("Options")
        table.add_row("Ingest", "1) Add URLs")
        table.add_row("Q&A", "2) Ask questions")
        table.add_row("Inspect", "3) Info")
        table.add_row("Visualize", "4) Viz docs   5) Viz metrics")
        table.add_row("Evaluate", "6) Eval")
        table.add_row("Sources", "7) Manage sources (list/delete/clear)")
        table.add_row("Exit", "0) Exit")

        console.print(table)

        choice = _prompt_menu_choice("\n[bold]Select:[/bold] ", valid=set("01234567"))

        if choice == "0":
            console.print("[dim]Goodbye.[/dim]")
            return

        if choice == "1":
            raw = console.input(
                "\nPaste one or more URLs (space or newline separated):\n> "
            ).strip()

            urls = [u.strip() for u in raw.replace("\n", " ").split(" ") if u.strip()]

            if not urls:
                console.print("[yellow]No URLs provided.[/yellow]")
                continue

            add(urls)

            continue

        if choice == "2":
            ask()
            continue

        if choice == "3":
            info()
            continue

        if choice == "4":
            _viz_docs(Path("data/visuals/cortex_docs.png"), show=True)
            continue

        if choice == "5":
            _viz_metrics(Path("data/visuals/cortex_metrics.png"), show=True)
            continue

        if choice == "6":
            name = (
                typer.prompt("Experiment name", default="experiment").strip()
                or "experiment"
            )

            q_path = (
                typer.prompt(
                    "Questions JSON path", default="data/eval/questions.json"
                ).strip()
                or "data/eval/questions.json"
            )

            save = typer.confirm("Save results JSON?", default=True)

            if not _ensure_questions_file(Path(q_path)):
                console.print("[dim]Aborted.[/dim]")
                continue
            try:
                evaluate(questions_file=Path(q_path), name=name, save=save)
            except Exception as e:
                console.print(f"[red]Eval failed:[/red] {e}")

            continue

        if choice == "7":
            _run_sources_menu()
            continue


def _run_sources_menu() -> None:
    from .store import collection_exists, delete_source, list_sources

    if not collection_exists():
        console.print("[red]No knowledge base found.[/red] Run 'cortex add' first.")
        return

    questions_path = Path("data/eval/questions.json")

    while True:
        console.print(
            Panel(
                "[bold]Sources[/bold]\n[dim]Manage ingested sources.[/dim]",
                border_style="cyan",
            )
        )

        console.print("[bold]1)[/bold] List sources")
        console.print("[bold]2)[/bold] Delete source")
        console.print("[bold]3)[/bold] Clear all")
        console.print("[bold]4)[/bold] Add source")
        console.print("[bold]5)[/bold] Create eval questions (missing only)")
        console.print("[bold]0)[/bold] Back")

        choice = _prompt_menu_choice(
            "\n[bold]Select:[/bold] ", valid={"0", "1", "2", "3", "4", "5"}
        )

        if choice == "0":
            return

        if choice == "1":
            sources = list_sources()

            if not sources:
                console.print("[yellow]No sources found.[/yellow]")
                continue

            questions = _load_questions(questions_path)

            table = Table(show_header=True, header_style="bold cyan")

            table.add_column("#", justify="right")
            table.add_column("Chunks", justify="right")
            table.add_column("Questions", justify="center")
            table.add_column("Domain", style="dim")
            table.add_column("Title")
            table.add_column("URL", style="dim")

            for i, s in enumerate(sources, start=1):
                has_q = (
                    "?"
                    if questions is None
                    else (
                        "yes" if _source_has_questions(questions, s["source"]) else "no"
                    )
                )

                table.add_row(
                    str(i),
                    str(s["count"]),
                    has_q,
                    s.get("domain", ""),
                    s.get("title", ""),
                    s["source"],
                )

            console.print(table)

            continue

        if choice == "2":
            sources = list_sources()

            if not sources:
                console.print("[yellow]No sources found.[/yellow]")
                continue

            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", justify="right")
            table.add_column("Chunks", justify="right")
            table.add_column("Title")
            table.add_column("URL", style="dim")

            for i, s in enumerate(sources, start=1):
                table.add_row(str(i), str(s["count"]), s.get("title", ""), s["source"])

            console.print(table)

            raw = typer.prompt("Pick a source number to delete", default="0")

            try:
                idx = int(raw)
            except ValueError:
                console.print("[red]Invalid number.[/red]")
                continue

            if idx <= 0 or idx > len(sources):
                console.print("[dim]Aborted.[/dim]")
                continue

            target = sources[idx - 1]["source"]

            if not typer.confirm(
                f"Delete source?\n{target}\nThis cannot be undone.", default=False
            ):
                console.print("[dim]Aborted.[/dim]")
                continue

            deleted = delete_source(target)
            console.print(f"[green]Deleted {deleted} chunks.[/green]")

            continue

        if choice == "3":
            clear()
            continue

        if choice == "4":
            raw = console.input(
                "\nPaste one or more URLs (space or newline separated):\n> "
            ).strip()

            urls = [u.strip() for u in raw.replace("\n", " ").split(" ") if u.strip()]

            if not urls:
                console.print("[yellow]No URLs provided.[/yellow]")
                continue

            add(urls)

            # After adding, ask whether to create questions per-source (only if missing).
            sources_now = {s["source"] for s in list_sources()}

            for url in urls:
                if url not in sources_now:
                    continue

                questions = _load_questions(questions_path)

                if questions is not None and _source_has_questions(questions, url):
                    continue

                if typer.confirm(
                    f"Create eval questions now for this source?\n{url}", default=True
                ):
                    _create_questions_for_source(questions_path, url)

            continue

        if choice == "5":
            sources = list_sources()

            if not sources:
                console.print("[yellow]No sources found.[/yellow]")
                continue

            questions = _load_questions(questions_path)

            if questions is None:
                console.print(
                    f"[yellow]Questions file is invalid:[/yellow] {questions_path}"
                )

                if not typer.confirm(
                    "Overwrite with a valid file when creating questions?", default=True
                ):
                    console.print("[dim]Aborted.[/dim]")
                    continue

                questions = []

            missing = _sources_missing_questions(sources, questions)

            if not missing:
                console.print("[green]All sources already have questions.[/green]")
                continue

            table = Table(show_header=True, header_style="bold cyan")

            table.add_column("#", justify="right")
            table.add_column("Chunks", justify="right")
            table.add_column("Title")
            table.add_column("URL", style="dim")

            for i, s in enumerate(missing, start=1):
                table.add_row(str(i), str(s["count"]), s.get("title", ""), s["source"])

            console.print(table)

            raw = typer.prompt(
                "Pick a source number to create questions for", default="0"
            )

            try:
                idx = int(raw)
            except ValueError:
                console.print("[red]Invalid number.[/red]")
                continue

            if idx <= 0 or idx > len(missing):
                console.print("[dim]Aborted.[/dim]")
                continue

            target = missing[idx - 1]["source"]
            _create_questions_for_source(questions_path, target)

            continue


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """
    Default entrypoint: if no subcommand was provided, run the interactive menu.
    """

    if ctx.invoked_subcommand is None:
        _run_menu()


@app.command()
def add(
    urls: list[str] = typer.Argument(..., help="One or more article URLs to ingest."),
) -> None:
    """
    Scrape and ingest articles into the knowledge base.
    """

    import time

    from .chunker import TextChunker
    from .config import config
    from .logging_utils import get_logger
    from .scraper import ScraperError, scrape_article
    from .store import upsert_chunks

    log = get_logger(__name__)

    if not (config.allowed_domains or []):
        console.print(
            "[red]Scraping is locked down.[/red] "
            "Set [bold]ALLOWED_DOMAINS[/bold] in your .env (comma-separated)."
        )

        raise typer.Exit(2)

    chunker = TextChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
    total_chunks = 0

    for url in urls:
        console.print(f"\n[cyan]→[/cyan] Fetching: [dim]{url}[/dim]")

        try:
            article = scrape_article(url)
        except ScraperError as e:
            console.print(f"  [red]✗[/red] Failed: {e}")
            continue

        console.print(f"[dim]{article.title}[/dim]")

        log.info(
            "Fetched %s (title=%s, chars=%d)", url, article.title, len(article.content)
        )

        t_chunk0 = time.perf_counter()

        with console.status("[dim]Chunking article text...[/dim]", spinner="dots"):
            chunks = chunker.chunk(
                article.content,
                metadata={
                    "source": url,
                    "title": article.title,
                    "domain": article.domain,
                },
            )

        log.info(
            "Chunked %s into %d chunks in %.2fs",
            url,
            len(chunks),
            time.perf_counter() - t_chunk0,
        )

        if not chunks:
            console.print("[yellow]⚠[/yellow] No chunks extracted — skipping")
            continue

        with console.status(
            "[dim]Embedding & saving to knowledge base...[/dim]", spinner="dots"
        ):
            upsert_chunks(chunks)

        total_chunks += len(chunks)

        console.print(f"[green]✓[/green] {len(chunks)} chunks ingested")

    console.print(
        Panel(
            f"[bold green]Done.[/bold green] Added {total_chunks} chunks to knowledge base.",
            border_style="green",
        )
    )


@app.command()
def ask() -> None:
    """
    Start an interactive Q&A session with your knowledge base.
    """

    from .generator import generate_answer
    from .retriever import retrieve
    from .store import collection_exists

    if not collection_exists():
        console.print(
            "[red]No knowledge base found.[/red] Run [bold]cortex add <url>[/bold] first."
        )

        raise typer.Exit(1)

    console.print(
        Panel(
            "[bold cyan]Cortex — Knowledge Worker[/bold cyan]\n"
            "Ask questions about your ingested articles.\n"
            "[dim]Type 'exit' or press Ctrl+C to quit.[/dim]",
            border_style="cyan",
        )
    )

    while True:
        try:
            question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        with console.status("[dim]Searching knowledge base...[/dim]", spinner="dots"):
            results = retrieve(question)

        if not results:
            console.print("[yellow]No relevant content found.[/yellow]")
            continue

        context = _format_retrieval_context(results)

        with console.status("[dim]Generating answer...[/dim]", spinner="dots"):
            answer = generate_answer(question, context)

        console.print(
            Panel(
                answer,
                title="[bold]Answer[/bold]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Show sources (deduplicated, max 3)
        sources = list(dict.fromkeys(r.source for r in results))[:3]

        if sources:
            console.print("[dim]Sources:[/dim]")

            for s in sources:
                console.print(f"  [dim]• {s}[/dim]")


@app.command()
def clear() -> None:
    """
    Delete the knowledge base. You will need to re-run 'cortex add'.
    """

    import shutil

    from .config import config
    from .store import count, reset_store

    if not config.chroma_path.exists():
        console.print("[dim]No knowledge base to clear.[/dim]")
        return

    n = count()

    confirm = typer.confirm(
        f"Delete knowledge base ({n} chunks)? This cannot be undone.",
        default=False,
    )

    if confirm:
        # Ensure we don't keep a stale, open handle to a deleted store.
        reset_store()
        shutil.rmtree(config.chroma_path)
        console.print("[green]Knowledge base cleared.[/green]")
    else:
        console.print("[dim]Aborted.[/dim]")


@app.command("eval")
def evaluate(
    questions_file: Path = typer.Option(
        Path("data/eval/questions.json"),
        "--questions",
        "-q",
        help="Path to evaluation questions JSON file.",
        show_default=True,
    ),
    name: str = typer.Option(
        "experiment",
        "--name",
        "-n",
        help="Label for this evaluation run (used in saved report filename).",
    ),
    save: bool = typer.Option(True, help="Save results JSON to data/eval/results/."),
) -> None:
    """
    Run retrieval evaluation and report MRR and Hit Rate.
    """

    import json

    from json import JSONDecodeError
    from .config import config
    from .evaluator import EvalQuery, EvalReport, EvalResult
    from .retriever import retrieve

    if not questions_file.exists():
        console.print(f"[red]File not found:[/red] {questions_file}")
        console.print("Create evaluation questions in data/eval/questions.json")

        raise typer.Exit(1)

    try:
        with questions_file.open(encoding="utf-8") as f:
            raw_queries = json.load(f)
    except JSONDecodeError as e:
        console.print(f"[red]Invalid JSON:[/red] {questions_file}")
        console.print(f"[dim]{e}[/dim]")

        raise typer.Exit(1)

    queries = [EvalQuery(**q) for q in raw_queries]
    results: list[EvalResult] = []

    # track() wraps an iterable and displays a progress bar
    for query in track(queries, description="Evaluating retrieval..."):
        chunks = retrieve(query.question)
        sources = [c.source for c in chunks]

        results.append(
            EvalResult(
                query_id=query.id,
                question=query.question,
                retrieved_sources=sources,
                relevant_source=query.relevant_source,
            )
        )

    report = EvalReport(
        experiment_name=name,
        results=results,
        config_snapshot={
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "top_k": config.top_k,
            "query_expansion": config.use_query_expansion,
            "embedding_model": config.embedding_model,
        },
    )

    # Display results table
    table = Table(show_header=True, header_style="bold cyan")

    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="center")
    table.add_column("Benchmark", style="dim")

    mrr = report.mrr
    hit = report.hit_rate

    mrr_color = "green" if mrr >= 0.6 else ("yellow" if mrr >= 0.3 else "red")
    hr_color = "green" if hit >= 0.7 else ("yellow" if hit >= 0.4 else "red")

    table.add_row(
        "MRR", f"[{mrr_color}]{mrr:.4f}[/{mrr_color}]", "≥0.6 good / ≥0.8 excellent"
    )

    table.add_row(
        "Hit Rate", f"[{hr_color}]{hit:.4f}[/{hr_color}]", "≥0.7 good / ≥0.9 excellent"
    )

    table.add_row("Queries evaluated", str(len(results)), "")

    console.print(
        Panel(table, title=f"[bold]Evaluation: {name}[/bold]", border_style="cyan")
    )

    if save:
        results_dir = Path("data/eval/results")
        out_path = results_dir / f"eval_{name}.json"
        report.save(out_path)
        console.print(f"[dim]Saved: {out_path}[/dim]")


@app.command()
def viz(
    what: str = typer.Argument(..., help="What to visualize: 'docs' or 'metrics'"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output PNG path."
    ),
    show: bool = typer.Option(
        True,
        "--show/--no-show",
        help="Render the generated image in the terminal when supported.",
    ),
) -> None:
    """
    Generate visualizations: document map (t-SNE) or metrics comparison.
    """

    if what == "docs":
        _viz_docs(output or Path("data/visuals/cortex_docs.png"), show=show)
    elif what == "metrics":
        _viz_metrics(output or Path("data/visuals/cortex_metrics.png"), show=show)
    else:
        console.print(
            f"[red]Unknown option:[/red] '{what}'. Choose 'docs' or 'metrics'."
        )

        raise typer.Exit(1)


def _try_render_image_in_terminal(path: Path) -> None:
    if not console.is_terminal:
        return

    try:
        # `rich` itself doesn't ship image rendering; use rich-pixels if installed.
        from PIL import Image  # type: ignore[import-not-found]
        from rich_pixels import Pixels  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        with Image.open(path) as im:
            # Keep a sensible width for typical terminals.
            console.print(Pixels.from_image(im, width=100))
    except Exception:
        # Rendering is best-effort; path output is still printed by caller.
        return


def _viz_docs(output: Path, show: bool) -> None:
    from .store import collection_exists, get_all_for_visualization
    from .visualizer import visualize_documents

    if not collection_exists():
        console.print("[red]No knowledge base found.[/red] Run 'cortex add' first.")
        raise typer.Exit(1)

    with console.status("[dim]Loading embeddings...[/dim]"):
        embeddings, metadatas = get_all_for_visualization()

    console.print(f"[dim]Computing t-SNE for {len(embeddings)} chunks...[/dim]")

    with console.status("[dim]Rendering t-SNE (this may take 20-60s)...[/dim]"):
        path = visualize_documents(embeddings, metadatas, output)

    console.print(f"[green]✓[/green] Saved: [bold]{path}[/bold]")

    if show:
        _try_render_image_in_terminal(Path(path))


def _viz_metrics(output: Path, show: bool) -> None:
    from .evaluator import EvalReport
    from .visualizer import visualize_metrics

    results_dir = Path("data/eval/results")
    reports = EvalReport.load_all(results_dir)

    if not reports:
        console.print(
            "[yellow]No evaluation results found.[/yellow] Run 'cortex eval' first."
        )

        return

    path = visualize_metrics(reports, output)
    console.print(f"[green]✓[/green] Saved: [bold]{path}[/bold]")

    if show:
        _try_render_image_in_terminal(Path(path))


@app.command()
def info() -> None:
    """
    Display information about the current knowledge base.
    """

    from .config import config
    from .store import count

    if not config.chroma_path.exists():
        console.print("[dim]No knowledge base found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold cyan")

    table.add_column("Property")
    table.add_column("Value")
    table.add_row("Chunks stored", str(count()))
    table.add_row("Embedding model", config.embedding_model)
    table.add_row("Generation model", config.generation_model)
    table.add_row("Chunk size", str(config.chunk_size))
    table.add_row("Chunk overlap", str(config.chunk_overlap))
    table.add_row("Top-K results", str(config.top_k))

    table.add_row(
        "Query expansion", "enabled" if config.use_query_expansion else "disabled"
    )
    
    table.add_row("Store path", str(config.chroma_path.absolute()))

    console.print(table)


## NOTE: `clear()` command is defined earlier in this file.
