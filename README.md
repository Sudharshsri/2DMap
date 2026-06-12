# Video to 2D Floor Plan Generator (2DMap)

This project provides an automated pipeline to generate 2D floor plans (in AutoCAD DXF and PNG formats) directly from a video walkthrough of an indoor space. It uses local Vision-Language Models (VLMs) and Large Language Models (LLMs) via Ollama to analyze video frames, extract architectural features, and synthesize a structured layout that adheres to IS 962 architectural drafting standards.

## How It Works

The system operates in a multi-step pipeline:

1. **Step 0: Context Extraction (`step0_parse_is_962.py`)**
   Reads the Indian Standard (IS 962) Code of Practice for Architectural and Building Drawings PDF to extract drafting rules regarding line types, sheet sizes, lettering, and graphical symbols.

2. **Step 1: Frame Extraction (`step1_extract_frames.py`)**
   Processes the input video (`input/room_video.mp4`) and extracts evenly spaced frames to act as input for the Vision model.

3. **Step 2A: Vision Analysis (`step2a_extract_observations.py`)**
   Analyzes each extracted frame using the `moondream` vision model to identify rooms, dimensions, walls, doors, windows, and spatial relationships. The output is a series of textual observations.

4. **Step 2B: Layout Synthesis (`step2b_generate_json.py`)**
   Uses the `llama3.2:3b` text model to process all the text observations along with the IS 962 rules and synthesize them into a single, cohesive JSON representation of the floor plan layout with logical coordinates.

5. **Step 3: DXF & Image Generation (`step3_generate_dxf.py`)**
   Reads the generated JSON and uses `ezdxf` and `matplotlib` to render the floor plan as a standard AutoCAD DXF file and a PNG preview.

## Prerequisites
Step 1
- Python 3.11.9

Step 2
- Create a virtual environment
```bash
py -3.11 -m venv venv
```

Step 3
- [Ollama](https://ollama.com/) installed and running locally.
### Required Ollama Models
You need to pull the required local models before running the scripts:

```bash
ollama pull moondream
```

```bash
ollama pull llama3.2:3b
```
Step 4
### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Directory Structure

```text
2DMap/
├── input/
│   ├── IS 962.pdf         # Standard drafting rules PDF
│   └── room_video.mp4     # Input walkthrough video
├── output/                # Generated artifacts will be saved here
│   ├── frames/            # Extracted video frames (.jpg)
│   ├── is962_context.txt  # Parsed rules text
│   ├── observations.txt   # Frame-by-frame VLM descriptions
│   ├── floor_plan.json    # Synthesized structural layout
│   ├── floor_plan.dxf     # Final AutoCAD file
│   └── floor_plan.png     # Final visual preview
├── requirements.txt       # Python dependencies
├── step0_parse_is_962.py
├── step1_extract_frames.py
├── step2a_extract_observations.py
├── step2b_generate_json.py
└── step3_generate_dxf.py
```

## Usage

Place your source files in the `input` directory:
- Your architectural standard reference: `input/IS 962.pdf`
- Your source video: `input/room_video.mp4`

Run the pipeline sequentially:

```bash
# 1. Parse the IS 962 PDF
python step0_parse_is_962.py

# 2. Extract frames from the video
python step1_extract_frames.py

# 3. Analyze frames with the Vision model (moondream)
python step2a_extract_observations.py

# 4. Synthesize the floor plan JSON (llama3.2:3b)
python step2b_generate_json.py

# 5. Generate the final DXF and PNG files
python step3_generate_dxf.py
```

The final files `floor_plan.dxf` and `floor_plan.png` will be located in the `output/` directory.
