"""Run local semantic retrieval and optionally an Ollama answer from CMD."""
import argparse
from pathlib import Path
from app.embeddings import LocalEmbedder
from app.vector_store import PersistentRetriever
from app.ollama_client import generate_answer,LLMUnavailable

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k",type=int,default=3)
    parser.add_argument("--search-only",action="store_true", help="Skip local LLM")
    args=parser.parse_args()
    retriever=PersistentRetriever(ROOT/"data/chroma_db",LocalEmbedder())
    matches=retriever.search(args.query,args.top_k)
    print("\nRETRIEVED DOCUMENTS")
    for n,m in enumerate(matches,1):
        print(f"[{n}] {m.source} page {m.page or '-'} cosine similarity {m.score:.3f}\n{m.text[:450]}\n")
    if not args.search_only:
        try: print("\nANSWER\n"+generate_answer(args.query,matches))
        except LLMUnavailable as exc: print("Generation unavailable:", exc)


if __name__=="__main__": main()
