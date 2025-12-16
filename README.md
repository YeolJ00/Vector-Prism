# Vector Prism 🎨✨

Vector Prism is a small toolkit for generating expressive SVG animations by combining visual-language models (VLMs) and language models (LLMs) with pragmatic SVG parsing and composition utilities.

Key ideas:
- Decompose SVGs into meaningful elements and semantic classes
- Use LLM/VLM prompts to create animation plans and animation code (HTML/CSS)
- Export per-frame vector assets or raster captures for downstream rendering or evaluation

---

## 🚀 Features

- SVG decomposition and semantic tagging
- LLM-driven animation planning
- LLM-driven CSS generation for animations
- Utilities for exporting frames to PDF/PNG and saving videos
- Simple CLI entry point for driving experiments and debugging

---

## 🧰 Installation

Clone the repository and install the dependencies listed in `requirements.txt` (preferably inside a virtual environment):

```bash
git clone https://github.com/YeolJ00/vector-prism.git
cd vector-prism
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- Some utilities rely on Selenium and a compatible Chromium/Chrome binary for exporting frames; see `utils/export_frames.py` for options.
- Optional ML components (ViCLIP, DOVER) will require GPU drivers and model checkpoints to run efficiently.

---

## ▶️ Usage

Basic usage via the main CLI:

```bash
python main.py --exp_name demo --test_json svg/test.jsonl --test_plan_json logs/plans.jsonl
```

Interactive helpers will prompt you to pick an SVG and an instruction from `svg/test.jsonl`.

Export frames from an HTML animation (vector PDF or raster PNG):

```bash
python utils/export_frames.py --input_file path/to/animation.html --fps 24 --duration 5 --format pdf
```

---

## 🧩 Project layout

- `main.py` — CLI entry point that wires up the models, planner, parser, and generator
- `svg_decomposition.py` — SVG parsing, semantic tagging, and VLM-driven analysis
- `animation_planner.py` — Creates animation plans (JSON) using prompts
- `animation_generator.py` — Generates CSS/HTML snippets from plans
- `svg_composition.py` — Utilities to restructure and group SVG elements
- `utils/` — Helper utilities (export frames, metrics, setup/CLI helpers)

---

## 💡 Development notes

- Code is structured for importability; modules avoid side-effectful top-level execution. Run `main.py` to drive full experiments.
- Logging is used throughout — configure `--exp_name` to create a timestamped `logs/` directory.
- Tests: none included yet; contributions adding unit tests are welcome.

---

## 🤝 Contributing

Contributions are welcome! Please open issues for bugs or feature ideas and submit PRs that include tests and clear descriptions.

---

## 📄 License

MIT — see `LICENSE` (add one if you wish to license the project).

---

If you'd like, I can also generate example `svg/test.jsonl` entries, a small example HTML output, and/or add a short developer guide for running and debugging locally. ✅