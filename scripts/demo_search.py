"""Run local semantic search without starting a server."""
import argparse
from app.main import get_index


def main():
    parser = argparse.ArgumentParser(description="Find relevant local document passages")
    parser.add_argument("--query", default="How can we check whether a forecasting model is reliable?")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    index = get_index()
    for rank, hit in enumerate(index.search(args.query, args.top_k), 1):
        print(f"\n#{rank}  {hit.source}  cosine={hit.score:.3f}  id={hit.chunk_id}\n{hit.text}\n")


if __name__ == "__main__":
    main()
