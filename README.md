# Vector Prism

Official repository for the Vector Prism paper. This project implements the pipeline used in the paper to animate SVGs: SVG input → semantic parsing → LLM/VLM-driven planning → CSS/HTML generation.


## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline (interactive):

```bash
python main.py --exp_name myrun --test_json svg/test.jsonl --test_plan_json logs/plans.jsonl
```

Export per-frame PDFs/PNGs (headless Chrome required):

```bash
python utils/export_frames.py --input_file path/to/animation.html --fps 24 --duration 5 --format pdf
```


## Project layout

- `main.py` — pipeline entry point used in experiments
- `svg_decomposition.py` — parser and semantic tagging
- `animation_planner.py` — LLM-based plan generation
- `animation_generator.py` — CSS/HTML generation
- `svg_composition.py` — grouping/restructuring utilities
- `utils/` — export, metrics, and setup helpers
