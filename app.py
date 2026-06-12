"""CoreForge-UI: lightweight local AI finetuning dashboard."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import List

import gradio as gr

from components.dataset_gen import DatasetSynthesizer
from components.quantizer import ModelCompiler
from components.trainer import BASE_MODELS, LocalTrainer, LocalTrainerConfig
from utils.helpers import ensure_directory


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = ensure_directory(BASE_DIR / "data")
OUTPUT_DIR = ensure_directory(BASE_DIR / "outputs")


def _read_dataset_preview(dataset_path: str | None, rows: int = 8) -> List[dict]:
    if not dataset_path:
        return []
    path = Path(dataset_path)
    if not path.exists():
        return []

    output: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(output) >= rows:
                break
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            output.append(parsed)
    return output


def _coerce_dataset_path(dataset_path: str | None, dataset_state: str | None) -> str:
    return (dataset_path or "").strip() or (dataset_state or "").strip()


def _coerce_gguf_path(gguf_path: str | None, gguf_state: str | None) -> str:
    return (gguf_path or "").strip() or (gguf_state or "").strip()


def _normalize_base_model(selected: str) -> str:
    if not selected:
        return "mistral-7b"
    return selected


def generate_dataset(
    source_file,
    chunk_size: int,
    chunk_overlap: int,
    max_examples: int,
    use_ollama: bool,
):
    if source_file is None:
        return "Please upload a .pdf or .txt file.", [], "", ""

    source_path = getattr(source_file, "name", str(source_file))
    try:
        engine = DatasetSynthesizer(output_dir=DATA_DIR, use_ollama=use_ollama)
        result = engine.generate(
            source_file=source_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_examples=max_examples,
        )
        dataset_path = result["dataset_path"]
        preview = _read_dataset_preview(dataset_path, rows=10)
        message = (
            "Dataset generated successfully.\n"
            f"Source: {result['source_path']}\n"
            f"Chunks: {result['chunks']}\n"
            f"Records: {result['records']}\n"
            f"Path: {dataset_path}"
        )
        return message, preview, dataset_path, dataset_path
    except Exception as exc:
        return f"Dataset generation failed: {exc}", [], "", ""


def _train_worker(
    trainer: LocalTrainer,
    config: LocalTrainerConfig,
    out_queue: queue.Queue,
):
    try:
        result = trainer.train(
            config=config,
            on_log=lambda message: out_queue.put(("log", message)),
        )
        out_queue.put(("done", result["adapter_dir"]))
    except Exception as exc:
        out_queue.put(("error", str(exc)))


def run_training(
    base_model: str,
    dataset_path: str,
    dataset_state: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    lora_r: int,
    lora_alpha: int,
):
    selected_dataset = _coerce_dataset_path(dataset_path, dataset_state)
    if not selected_dataset:
        yield "Please generate/select a dataset path first.", "", ""
        return

    cfg = LocalTrainerConfig(
        base_model=_normalize_base_model(base_model),
        dataset_path=selected_dataset,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    logs: List[str] = []
    out_q: queue.Queue = queue.Queue()
    trainer = LocalTrainer(output_root=OUTPUT_DIR)
    thread = threading.Thread(target=_train_worker, args=(trainer, cfg, out_q), daemon=True)
    thread.start()

    while True:
        kind, payload = out_q.get()
        if kind == "log":
            logs.append(payload)
            yield "\n".join(logs), "", ""
        elif kind == "done":
            logs.append(f"Adapter saved: {payload}")
            yield "\n".join(logs), payload, payload
            break
        elif kind == "error":
            logs.append(f"ERROR: {payload}")
            yield "\n".join(logs), "", ""
            break


def compile_adapter(
    base_model: str,
    adapter_dir: str,
    adapter_state: str,
    quant_format: str,
):
    adapter_dir = _coerce_dataset_path(adapter_dir, adapter_state)
    if not adapter_dir:
        return "Please run training first or choose a trained adapter directory.", "", "", ""

    try:
        compiler = ModelCompiler(output_dir=OUTPUT_DIR)
        compile_result = compiler.compile_to_gguf(
            base_model=_normalize_base_model(base_model),
            adapter_dir=adapter_dir,
            quantization=quant_format,
            workspace=OUTPUT_DIR / Path(adapter_dir).name,
        )
        modelfile = compiler.write_modelfile(compile_result["gguf_path"])
        return (
            "Compilation succeeded.\n" + compile_result["stdout"],
            compile_result["gguf_path"],
            str(modelfile),
            compile_result["gguf_path"],
        )
    except Exception as exc:
        return f"Compilation failed: {exc}", "", "", _coerce_dataset_path(adapter_dir, "")


def deploy_to_ollama(
    gguf_path: str,
    gguf_state: str,
    model_name: str,
    base_model: str,
):
    del base_model  # Reserved for future extension.
    selected_gguf = _coerce_gguf_path(gguf_path, gguf_state)
    if not selected_gguf:
        return "Please compile a GGUF artifact first."
    try:
        compiler = ModelCompiler(output_dir=OUTPUT_DIR)
        modelfile = compiler.write_modelfile(selected_gguf, model_alias=model_name)
        return compiler.deploy_to_ollama(modelfile, model_name=model_name)
    except Exception as exc:
        return f"Deploy to Ollama failed: {exc}"


def build_ui():
    css = """
    .main {background-color: #0b1220;}
    .gradio-container {background-color: #0b1220; color: #dbeafe;}
    .gradio-container .gr-button {min-height: 44px;}
    """
    with gr.Blocks(title="CoreForge-UI", theme=gr.themes.Base(), css=css) as app:
        gr.Markdown("# CoreForge-UI")
        gr.Markdown(
            "A streamlined local fine-tuning interface for text-to-instruction "
            "workflows powered by QLoRA and local deployment tooling."
        )

        dataset_path_state = gr.State(value="")
        adapter_path_state = gr.State(value="")
        gguf_path_state = gr.State(value="")

        with gr.Tabs():
            with gr.TabItem("Dataset Forge"):
                with gr.Row():
                    source_file = gr.File(
                        label="Upload PDF or Text Source",
                        file_types=[".pdf", ".txt", ".md", ".csv", ".json", ".log"],
                    )
                    with gr.Column(scale=1):
                        chunk_size = gr.Slider(250, 2500, value=1200, step=50, label="Chunk Size")
                        chunk_overlap = gr.Slider(0, 300, value=120, step=10, label="Chunk Overlap")
                        max_examples = gr.Number(
                            value=120,
                            minimum=10,
                            maximum=500,
                            step=10,
                            label="Max examples to keep",
                        )
                        use_ollama = gr.Checkbox(label="Use local Ollama for synthetic generation", value=False)
                        generate_btn = gr.Button("Generate Synthetic Dataset", variant="primary")
                generate_status = gr.Textbox(lines=8, label="Dataset Forge Log")
                dataset_path_box = gr.Textbox(label="Generated Dataset Path", interactive=False)
                preview_table = gr.Dataframe(headers=["instruction", "input", "output"], interactive=False)

                generate_btn.click(
                    generate_dataset,
                    inputs=[source_file, chunk_size, chunk_overlap, max_examples, use_ollama],
                    outputs=[generate_status, preview_table, dataset_path_box, dataset_path_state],
                )

                dataset_path_state.change(
                    lambda path: path,
                    inputs=[dataset_path_state],
                    outputs=[dataset_path_box],
                    show_progress=False,
                )

            with gr.TabItem("Fine-Tuning Hub"):
                with gr.Row():
                    base_model = gr.Dropdown(
                        choices=list(BASE_MODELS.keys()),
                        value="mistral-7b",
                        label="Base Model",
                    )
                    dataset_path_input = gr.Textbox(label="Dataset Path", placeholder="dataset path from Dataset Forge")
                with gr.Row():
                    learning_rate = gr.Number(value=2e-4, minimum=1e-6, maximum=1e-3, step=1e-6, label="Learning Rate")
                    batch_size = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                with gr.Row():
                    epochs = gr.Slider(1, 20, value=2, step=1, label="Epochs")
                    lora_r = gr.Slider(4, 128, value=16, step=4, label="LoRA Rank (r)")
                    lora_alpha = gr.Slider(16, 128, value=32, step=4, label="LoRA Alpha")

                start_btn = gr.Button("Start Training", variant="primary")
                train_log = gr.Textbox(lines=12, max_lines=20, label="Training Logs")
                adapter_path_box = gr.Textbox(label="Adapter Directory", interactive=False)

                dataset_path_state.change(
                    lambda path: path,
                    inputs=[dataset_path_state],
                    outputs=[dataset_path_input],
                    show_progress=False,
                )

                start_btn.click(
                    run_training,
                    inputs=[
                        base_model,
                        dataset_path_input,
                        dataset_path_state,
                        learning_rate,
                        batch_size,
                        epochs,
                        lora_r,
                        lora_alpha,
                    ],
                    outputs=[train_log, adapter_path_box, adapter_path_state],
                )

            with gr.TabItem("Edge Compilation"):
                with gr.Row():
                    compile_base_model = gr.Dropdown(
                        choices=list(BASE_MODELS.keys()),
                        value="mistral-7b",
                        label="Base Model",
                    )
                    quant_format = gr.Dropdown(
                        choices=["Q4_K_M", "Q4_0", "Q5_0", "Q8_0"],
                        value="Q4_K_M",
                        label="Quantization Format",
                    )
                with gr.Row():
                    adapter_dir_input = gr.Textbox(
                        label="Adapter Directory",
                        placeholder="Use output from training tab",
                    )
                    compile_btn = gr.Button("Compile GGUF", variant="primary")
                compile_log = gr.Textbox(lines=10, max_lines=18, label="Compilation Log")
                gguf_path_box = gr.Textbox(label="GGUF Path", interactive=False)
                modelfile_box = gr.Textbox(label="Modelfile", interactive=False)
                model_name = gr.Textbox(label="Ollama Model Name", value="coreforge-model")
                deploy_btn = gr.Button("Deploy to Ollama")
                deploy_status = gr.Textbox(label="Deploy Status")

                adapter_path_state.change(
                    lambda path: path,
                    inputs=[adapter_path_state],
                    outputs=[adapter_dir_input],
                    show_progress=False,
                )

                compile_btn.click(
                    compile_adapter,
                    inputs=[compile_base_model, adapter_dir_input, adapter_path_state, quant_format],
                    outputs=[compile_log, gguf_path_box, modelfile_box, gguf_path_state],
                )

                gguf_path_state.change(
                    lambda value: value,
                    inputs=[gguf_path_state],
                    outputs=[gguf_path_box],
                    show_progress=False,
                )

                deploy_btn.click(
                    deploy_to_ollama,
                    inputs=[gguf_path_box, gguf_path_state, model_name, compile_base_model],
                    outputs=deploy_status,
                )

        return app


if __name__ == "__main__":
    ui = build_ui()
    ui.launch(server_name="127.0.0.1", server_port=7860)
