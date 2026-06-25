# 2DMap

2DMap is an end-to-end indoor floor-plan generator that takes an indoor walkthrough video and automatically constructs a 2D floor plan, complete with room dimensions, doors, and transitions. It utilizes computer vision, vision-language models (VLMs), and large language models (LLMs) to intelligently analyze video frames and render a CAD-ready floor plan.

## Features

- **Automated Floor Plan Generation**: Converts a continuous video walkthrough into a structured 2D floor plan.
- **Multi-Stage Pipeline**: Modular design separating frame extraction, semantic perception, room segmentation, and CAD rendering.
- **Local AI Models**: Uses locally hosted models (`qwen2.5vl:3b` and `llama3.2:3b`) via [Ollama](https://ollama.com) for privacy and offline capabilities.
- **Smart Caching**: Each stage writes intermediate results to JSON, allowing you to skip previously completed stages and save time on subsequent runs.
- **Standardized Outputs**: Generates both an image preview (`.png`) and a CAD-compatible vector format (`.dxf`).

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally.

Before running the pipeline, you need to pull the required models using Ollama:
```bash
ollama pull qwen2.5vl:3b
ollama pull llama3.2:3b
```

## Installation

1. Clone or download the repository.
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your input video (e.g., `room_video.mp4`) in the `input/` directory and run the main script:

```bash
python run_pipeline.py --video input/room_video.mp4
```

### Command Line Arguments

- `--video`: Path to the input video file (default: `input/room_video.mp4`).
- `--output`: Directory to save the intermediate JSONs and final outputs (default: `output`).
- `--fps`: Frame extraction rate in frames per second (default: `0.5`).
- `--skip`: Stage numbers to skip by loading their cached JSON results. Example: `--skip 1 2` skips the first two stages.

### Pipeline Stages

1. **Frame extraction & motion heuristics**: Uses OpenCV to sample frames and calculate basic motion metrics.
2. **Per-frame semantic perception**: Analyzes each frame using `qwen2.5vl:3b` to identify room types, doors, and features. Includes a self-consistency audit step.
3. **Segment grouping & transition detection**: Groups consecutive similar frames into distinct "rooms" and identifies transitions (doors) between them.
4. **Floor-plan structuring**: Uses `llama3.2:3b` to reason about the spatial layout and generate the global floor plan structure.
5. **CAD rendering**: Uses `ezdxf` and `matplotlib` to render the final structured floor plan into `.dxf` and `.png` formats.

## Output

After a successful run, the `output/` directory will contain:
- `floor_plan.png`: A visual render of the floor plan.
- `floor_plan.dxf`: A CAD file of the floor plan.
- `stage*.json`: Intermediate cached data for each step of the pipeline.
- `frames/`: The extracted video frames used for analysis.
