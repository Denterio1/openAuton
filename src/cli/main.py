#!/usr/bin/env python3

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.agent import PrimeAgent, AgentConfig
from experience.episodes import EpisodeStore


def parse_args():
    parser = argparse.ArgumentParser(description="prime_agent — Self-improving AI engineer.")
    subparsers = parser.add_subparsers(dest="command")

    run_p = subparsers.add_parser("run", help="Run agent on a task.")
    run_p.add_argument("task", help="Task description.")
    run_p.add_argument("--verbose", "-v", action="store_true")

    train_p = subparsers.add_parser("train-file", help="Train on a file.")
    train_p.add_argument("file_path", help="Path to data file.")
    train_p.add_argument("--verbose", "-v", action="store_true")

    subparsers.add_parser("stats", help="Show statistics.")

    list_p = subparsers.add_parser("list", help="List episodes.")
    list_p.add_argument("--successful", "-s", action="store_true")
    list_p.add_argument("--limit", "-l", type=int, default=10)

    prune_p = subparsers.add_parser("prune", help="Prune old episodes.")
    prune_p.add_argument("--keep-recent", type=int, default=100)
    prune_p.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def run_task(task: str, verbose: bool):
    agent  = PrimeAgent(config=AgentConfig(verbose=verbose))
    result = agent.run(task)
    print(json.dumps(result, indent=2, default=str))


def run_on_file(file_path: str, verbose: bool):
    agent  = PrimeAgent(config=AgentConfig(verbose=verbose))
    result = agent.run_on_file(Path(file_path))
    print(json.dumps(result, indent=2, default=str))


def show_stats():
    store = EpisodeStore(Path("experiments/episodes"))
    stats = store.get_statistics()
    print(json.dumps(stats, indent=2, default=str))


def list_episodes(successful_only: bool, limit: int):
    store    = EpisodeStore(Path("experiments/episodes"))
    results  = store.search(
        success_only=successful_only,
        limit=limit,
    )
    if not results:
        print("No episodes found.")
        return
    for ep in results:
        print(
            f"[{ep.episode_id[:8]}] "
            f"{ep.task_type.value} | "
            f"{ep.status.value} | "
            f"acc={ep.evaluation.accuracy if ep.evaluation and ep.evaluation.accuracy else 'N/A'} | "
            f"{ep.task_description[:50]}"
        )


def prune_episodes(keep_recent: int, dry_run: bool):
    store     = EpisodeStore(Path("experiments/episodes"))
    to_delete = store.prune(keep_recent=keep_recent, dry_run=dry_run)
    if dry_run:
        print(f"Would delete {len(to_delete)} episodes.")
    else:
        print(f"Deleted {len(to_delete)} episodes.")


def main():
    args = parse_args()

    if args.command == "run":
        run_task(args.task, args.verbose)
    elif args.command == "train-file":
        run_on_file(args.file_path, args.verbose)
    elif args.command == "stats":
        show_stats()
    elif args.command == "list":
        list_episodes(args.successful, args.limit)
    elif args.command == "prune":
        prune_episodes(args.keep_recent, args.dry_run)
    else:
        print("Use --help for usage.")
        print("Examples:")
        print('  python src/cli/main.py run "Train a small transformer"')
        print("  python src/cli/main.py stats")
        print("  python src/cli/main.py list --limit 5")


if __name__ == "__main__":
    main()