"""Dataset synthesis engine for CoreForge-UI."""

from __future__ import annotations

import json
import random
import requests
from pathlib import Path
from typing import Any, Dict, List, Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from utils.helpers import ensure_directory, normalize_instructional_record, stable_slug, write_jsonl


class DatasetSynthesizer:
    """
    Build structured instruction datasets from source documents.
    """

    def __init__(
        self,
        output_dir: str | Path = "data",
        use_ollama: bool = False,
        ollama_url: str = "http://localhost:11434",
        hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    ) -> None:
        self.output_dir = ensure_directory(output_dir)
        self.use_ollama = use_ollama
        self.ollama_url = ollama_url.rstrip("/")
        self.hf_model_id = hf_model_id

    def _read_pdf(self, path: Path) -> str:
        try:
            reader = PdfReader(str(path))
            chunks: List[str] = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks)
        except Exception as exc:
            raise RuntimeError(f"Failed to read PDF {path}: {exc}") from exc

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            return path.read_bytes().decode("utf-8", errors="replace")
        except Exception as exc:
            raise RuntimeError(f"Failed to read text file {path}: {exc}") from exc

    def _load_document(self, file_path: str | Path) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix in {".txt", ".md", ".csv", ".json", ".log"}:
            return self._read_text(path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        return splitter.split_text(text)

    def _call_ollama(self, prompt: str) -> str:
        payload = {
            "model": "llama3",
            "stream": False,
            "prompt": prompt,
            "options": {"temperature": 0.3},
        }
        response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=45)
        response.raise_for_status()
        payload = response.json()
        text = payload.get("response") or ""
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text

    def _call_hf_pipeline(self, prompt: str) -> str:
        """
        Build a local HF pipeline call when a cached model is available.
        """
        from transformers import pipeline
        from transformers.pipelines.text_generation import TextGenerationPipeline

        generator = pipeline(
            "text-generation",
            model=self.hf_model_id,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.5,
        )
        if not isinstance(generator, TextGenerationPipeline):
            raise RuntimeError("Expected a text-generation pipeline object.")
        output = generator(prompt)
        if not output or not isinstance(output, Sequence):
            return ""
        text = output[0].get("generated_text", "")
        if not text:
            raise RuntimeError("HF pipeline returned an empty completion.")
        return str(text)

    def _synthesize_chunk(self, chunk: str) -> List[Dict[str, str]]:
        templates = [
            {
                "instruction": "Create a direct answer question from the context.",
                "input": "Context: " + chunk,
                "output": "Return a concise and faithful answer from the given context.",
            },
            {
                "instruction": "Generate a reasoning-heavy question that requires multi-step inference.",
                "input": "Context: " + chunk,
                "output": "Break down the reasoning process and then provide the final answer.",
            },
            {
                "instruction": "Explain a key concept from the context.",
                "input": "Context: " + chunk,
                "output": "Provide a simple but accurate explanation grounded in the source context.",
            },
        ]

        if self.use_ollama:
            prompt = (
                "Generate exactly three JSON objects as a JSON array. "
                "Each object must follow keys: instruction, input, output.\n\n"
                f"Context:\n{chunk}\n"
            )
            try:
                response = self._call_ollama(prompt)
                return self._parse_qa_response(response)
            except Exception:
                # Fall through to deterministic templates below.
                pass
        else:
            try:
                response = self._call_hf_pipeline(
                    f"Write three Alpaca JSON objects from this chunk with keys instruction/input/output: {chunk}"
                )
                parsed = self._parse_qa_response(response)
                if len(parsed) >= 1:
                    return parsed
            except Exception:
                # Fall through to deterministic templates below.
                pass

        return templates

    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        text = response.strip()
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            payload = text[start:end]
            parsed = json.loads(payload)
            if not isinstance(parsed, list):
                raise ValueError("Model did not return a list.")
            normalized: List[Dict[str, str]] = []
            for item in parsed:
                normalized.append(normalize_instructional_record(item))
            return normalized
        except Exception:
            # Fallback to a best-effort JSONL line parser.
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            normalized = []
            for line in lines:
                if line.startswith("{") and line.endswith("}"):
                    try:
                        normalized.append(normalize_instructional_record(json.loads(line)))
                    except Exception:
                        continue
            if normalized:
                return normalized
            raise

    def _trim_to_max_examples(self, rows: Sequence[Dict[str, str]], max_examples: int) -> List[Dict[str, str]]:
        sampled = list(rows)
        random.shuffle(sampled)
        return list(sampled[:max_examples])

    def generate(
        self,
        source_path: str | Path,
        chunk_size: int = 1200,
        chunk_overlap: int = 150,
        max_examples: int = 100,
    ) -> Dict[str, Any]:
        text = self._load_document(source_path)
        chunks = self._split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise RuntimeError("No readable text chunks were produced from the input document.")

        rows: List[Dict[str, str]] = []
        for chunk in chunks:
            for record in self._synthesize_chunk(chunk):
                try:
                    rows.append(normalize_instructional_record(record))
                except Exception:
                    continue

        if not rows:
            raise RuntimeError("No valid synthetic records were produced.")
        if max_examples > 0:
            rows = self._trim_to_max_examples(rows, max_examples=max_examples)

        filename = stable_slug(f"{Path(source_path).stem}-coreforge-alpaca") + ".jsonl"
        dataset_path = self.output_dir / filename
        write_jsonl(dataset_path, rows)

        return {
            "dataset_path": str(dataset_path.resolve()),
            "records": len(rows),
            "source_path": str(Path(source_path).resolve()),
            "chunks": len(chunks),
        }
