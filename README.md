<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/bJq6vjRRKv" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B-0626"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B-0626"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

**UPDATE 🤗(06/27)**: Dia is now available through [Hugging Face Transformers](https://github.com/huggingface/transformers)!

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B-0626). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B-0626). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/bJq6vjRRKv) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## Generation Guidelines

- Keep input text length moderate 
    - Short input (corresponding to under 5s of audio) will sound unnatural
    - Very long input (corresponding to over 20s of audio) will make the speech unnaturally fast.
- Use non-verbal tags sparingly, from the list in the README. Overusing or using unlisted non-verbals may cause weird artifacts.
- Always begin input text with `[S1]`, and always alternate between `[S1]` and `[S2]` (i.e. `[S1]`... `[S1]`... is not good)
- When using audio prompts (voice cloning), follow these instructions carefully:
    - Provide the transcript of the to-be cloned audio before the generation text.
    - Transcript must use `[S1]`, `[S2]` speaker tags correctly (i.e. single speaker: `[S1]`..., two speakers: `[S1]`... `[S2]`...)
    - Duration of the to-be cloned audio should be 5~10 seconds for the best results.
        (Keep in mind: 1 second ≈ 86 tokens)
- Put `[S1]` or `[S2]` (the second-to-last speaker's tag) at the end of the audio to improve audio quality at the end

## Quickstart

### Transformers Support

We now have a [Hugging Face Transformers](https://github.com/huggingface/transformers) implementation of Dia! You should install `main` branch of `transformers` to use it. See [hf.py](hf.py) for more information.

<details>
<summary>View more details</summary>

Install `main` branch of `transformers`

```bash
pip install git+https://github.com/huggingface/transformers.git
# or install with uv
uv pip install git+https://github.com/huggingface/transformers.git
```

Run `hf.py`. The file is as below.

```python
from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

text = [
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(
    **inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45
)

outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.mp3")
```

</details>

### Run with this repo

<details>
<summary> Install via pip </summary>

```bash
# Clone this repository
git clone https://github.com/nari-labs/dia.git
cd dia

# Optionally
python -m venv .venv && source .venv/bin/activate

# Install dia
pip install -e .
```

Or you can install without cloning.

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

Now, run some examples.

```bash
python example/simple.py
```
</details>


<details>
<summary>Install via uv</summary>

You need [uv](https://docs.astral.sh/uv/) to be installed.

```bash
# Clone this repository
git clone https://github.com/nari-labs/dia.git
cd dia
```

Run some examples directly.

```bash
uv run example/simple.py
```

</details>

<details>
<summary>Run Gradio UI</summary>

```bash
python app.py

# Or if you have uv installed
uv run app.py
```

</details>

<details>
<summary>Run with CLI</summary>

```bash
python cli.py --help

# Or if you have uv installed
uv run cli.py --help
```

</details>

> [!NOTE]
> The model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
> You can keep speaker consistency by either adding an audio prompt, or fixing the seed.

> [!IMPORTANT]
> If you are using 5000 series GPU, you should use torch 2.8 nightly. Look at the issue [#26](https://github.com/nari-labs/dia/issues/26) for more details.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.


## 💻 Hardware and Inference Speed

Dia has been tested on only GPUs (pytorch 2.0+, CUDA 12.6). CPU support is to be added soon.
The initial run will take longer as the Descript Audio Codec also needs to be downloaded.

These are the speed we benchmarked in RTX 4090.

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:-:|:-:|:-:|:-:|
| `bfloat16` | x2.1 | x1.5 | ~4.4GB |
| `float16` | x2.2 | x1.3 | ~4.4GB |
| `float32` | x1 | x0.9 | ~7.9GB |

We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support for ARM architecture and MacOS.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/bJq6vjRRKv) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>

# 🎬 Cartoon Animation RunPod Worker

[![Runpod](https://api.runpod.io/badge/elhalag93/cartoon-runpod)](https://console.runpod.io/hub/elhalag93/cartoon-runpod)

> Generate cartoon character animations with voice using Dia TTS and AnimateDiff on RunPod

## ✨ Features

- **Character Animation**: Generate high-quality animations using AnimateDiff with custom LoRA characters
- **Text-to-Speech**: Create realistic dialogue using Dia TTS model  
- **Combined Generation**: Produce animated videos with synchronized voice
- **RunPod Optimized**: Ready for deployment on RunPod serverless platform
- **Memory Efficient**: Optimized for various GPU configurations

## 🎭 Supported Characters

- **Temo**: Space explorer character with LoRA weights
- **Felfel**: Adventure character with LoRA weights

## 📋 API Reference

### Input Parameters

```json
{
  "input": {
    "task_type": "combined",
    "character": "temo", 
    "prompt": "character walking on moon surface, cartoon style",
    "dialogue_text": "[S1] Hello from the moon! [S2] What an adventure!",
    "num_frames": 16,
    "fps": 8,
    "guidance_scale": 7.5,
    "num_inference_steps": 15,
    "seed": 42,
    "max_new_tokens": 3072,
    "tts_guidance_scale": 3.0,
    "temperature": 1.8,
    "height": 512,
    "width": 512
  }
}
```

#### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `task_type` | Type of generation | `"animation"` | `"animation"`, `"tts"`, `"combined"` |
| `character` | Character to animate | `"temo"` | `"temo"`, `"felfel"` |
| `prompt` | Animation prompt | `""` | Any descriptive text |
| `dialogue_text` | Text for TTS generation | `""` | Text with `[S1]`, `[S2]` tags |

#### Animation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_frames` | Number of animation frames | `16` |
| `fps` | Frames per second | `8` |
| `guidance_scale` | How closely to follow prompt | `7.5` |
| `num_inference_steps` | Quality vs speed tradeoff | `15` |
| `seed` | Random seed for reproducibility | `null` |
| `height` | Output height in pixels | `512` |
| `width` | Output width in pixels | `512` |

#### TTS Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_new_tokens` | Maximum audio tokens to generate | `3072` |
| `tts_guidance_scale` | TTS guidance scale | `3.0` |
| `temperature` | TTS sampling temperature | `1.8` |

### Output Format

```json
{
  "task_type": "combined",
  "gif": "base64_encoded_gif_data",
  "mp4": "base64_encoded_mp4_data", 
  "audio": "base64_encoded_audio_data",
  "gif_path": "/workspace/outputs/temo_animation_1234567890.gif",
  "mp4_path": "/workspace/outputs/temo_animation_1234567890.mp4",
  "audio_path": "/workspace/temp/tts_output_1234567890.mp3",
  "seed": 42,
  "memory_usage": {
    "allocated_gb": 4.2,
    "total_gb": 24.0
  }
}
```

## 🚀 Quick Start

### 🌐 RunPod Deployment (Recommended)

1. **Use Pre-built Image**:
   ```
   your-dockerhub-username/cartoon-animation:latest
   ```

2. **Set Container Command**:
   - For worker mode: `python src/handler.py`
   - For web interface: `python launch.py web`
   - For API server: `python launch.py api`

3. **Configure Environment**:
   - Set `TEMO_LORA_URL` and `FELFEL_LORA_URL` for private LoRA weights
   - Expose port 7860 for web interface or 8000 for API

4. **Send Jobs**:
   ```json
   {
     "input": {
       "task_type": "combined",
       "character": "temo",
       "prompt": "temo character walking on moon surface",
       "dialogue_text": "[S1] Hello from the moon!",
       "num_frames": 16,
       "seed": 42
     }
   }
   ```

### 🎨 Interactive Web Interface (Local)

```bash
# Build and start the web interface
docker-compose up cartoon-web

# Access at: http://localhost:7860
```

### 🔌 API Server (Local)

```bash
# Build and start the API server
docker-compose up cartoon-api

# Access at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### 🧪 Test the System

```bash
# Run tests locally
python run_tests.py

# Test Docker build
docker build -t cartoon-animation-test .

# Test worker locally
python src/test_worker.py
```

## 🛠️ Project Structure

```
cartoon/
├── src/
│   ├── handler.py          # Main RunPod worker handler
│   ├── test_worker.py      # Local testing script
│   └── __init__.py
├── models/                 # Model files (add your models here)
│   ├── sdxl-turbo/
│   └── animatediff/
├── lora_models/           # Character LoRA weights
│   ├── temo_lora/
│   └── felfel_lora/
├── outputs/               # Generated animations
├── temp/                  # Temporary files
├── .runpod/
│   └── config.yaml        # RunPod configuration
├── Dockerfile             # Container definition
├── docker-compose.yml     # Local development
├── requirements.txt       # Python dependencies
├── test_input.json        # Sample input for testing
└── README.md
```

## 💾 Model Requirements

### Required Models
1. **SDXL Turbo**: Place in `models/sdxl-turbo/`
2. **AnimateDiff Motion Adapter**: Place in `models/animatediff/motion_adapter/`
3. **Character LoRAs**: Place in `lora_models/{character}_lora/`

### Automatic Downloads
- **Dia TTS Model**: Downloaded automatically from HuggingFace (`nari-labs/Dia-1.6B-0626`)

## 🎯 Usage Examples

### Animation Only
```json
{
  "input": {
    "task_type": "animation",
    "character": "temo",
    "prompt": "temo character dancing on moon surface",
    "num_frames": 24,
    "seed": 123
  }
}
```

### TTS Only
```json
{
  "input": {
    "task_type": "tts",
    "dialogue_text": "[S1] Welcome to the moon base! [S2] Ready for adventure?",
    "temperature": 1.5
  }
}
```

### Combined Generation
```json
{
  "input": {
    "task_type": "combined",
    "character": "felfel", 
    "prompt": "felfel character waving hello",
    "dialogue_text": "[S1] Hello everyone! [S2] Nice to meet you!"
  }
}
```

## 💻 System Requirements

### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti)
- **RAM**: 16GB system RAM
- **Storage**: 20GB free space

### Recommended Requirements  
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A6000)
- **RAM**: 32GB system RAM
- **Storage**: 50GB free space

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env` and configure:

```bash
RUNPOD_API_KEY=your_api_key_here
DIA_MODEL_CHECKPOINT=nari-labs/Dia-1.6B-0626
DEFAULT_NUM_FRAMES=16
DEFAULT_FPS=8
```

### Memory Optimization
The worker automatically applies memory optimizations:
- Sequential CPU offload
- Attention slicing
- VAE slicing and tiling
- Automatic garbage collection

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce `num_frames` (try 8-12)
- Reduce `num_inference_steps` (try 10-12)
- Lower resolution (`height: 480, width: 480`)

**Model Not Found**
- Ensure model files are in correct directories
- Check file permissions
- Verify LoRA weight file names

**Slow Generation**
- Use fewer inference steps
- Enable optimizations in Dockerfile
- Use smaller frame counts for testing

### Debug Mode
Enable debug logging:
```bash
export RUNPOD_DEBUG=true
python src/handler.py
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly  
5. Submit a pull request

## 🙏 Acknowledgments

- [Nari Labs](https://github.com/nari-labs/dia) for the Dia TTS model
- [Diffusers](https://github.com/huggingface/diffusers) for AnimateDiff implementation
- [RunPod](https://runpod.io) for serverless GPU infrastructure

---

**Ready to create amazing cartoon animations with voice on RunPod!** 🚀🎬🎵

# Cartoon Animation Worker for RunPod

> Generate animations with custom characters using SDXL, LoRA weights, and AnimateDiff as an endpoint on RunPod

## Features

- Animation generation using SDXL Turbo and AnimateDiff
- Custom character support with LoRA weights
- Automatic model loading and initialization
- RunPod endpoint for dynamic input handling

## API Reference

### Input Parameters

```json
{
  "input": {
    "character": "temo",
    "positive_prompt": "temo character walking on moon surface, detailed cartoon style",
    "negative_prompt": "blurry, low quality",
    "width": 512,
    "height": 512,
    "seed": 42,
    "num_frames": 16,
    "fps": 8,
    "guidance_scale": 7.5,
    "num_inference_steps": 15
  }
}
```

#### Core Parameters

| Parameter            | Description                                                            | Default |
| -------------------- | ---------------------------------------------------------------------- | ------- |
| character            | Character name for LoRA weights (e.g., 'temo', 'felfel')              | None    |
| positive_prompt      | Text description of what you want to generate                          | ""      |
| negative_prompt      | Text description of what you want to avoid in the generation           | ""      |
| width                | Output video width in pixels                                           | 512     |
| height               | Output video height in pixels                                          | 512     |
| seed                 | Random seed for reproducible results                                   | None    |
| num_frames           | Number of frames to generate                                           | 16      |
| fps                  | Frames per second for output video                                     | 8       |
| guidance_scale       | Classifier-free guidance scale (how closely to follow the prompt)      | 7.5     |
| num_inference_steps  | Number of denoising steps (higher = better quality, slower generation) | 15      |

## Deployment

Deploy this worker on RunPod using the GitHub Integration:

1. **Upload Models**: Ensure models are in place or set environment variables for download URLs.
2. **Push to GitHub**: `git push origin main` after committing changes.
3. **Deploy on RunPod**: Use RunPod.io, connect GitHub, select this repository, and choose a suitable GPU (e.g., RTX 4090).
4. **Test Deployment**: Use `test_input.json` for testing requests.

## Development

For local development:
- Run `setup_runpod.py` to prepare directories and check requirements.
- Use `runpod_animdiff.py` for local testing with hardcoded prompts.
- Deploy locally with Docker using `docker-compose.yml` for testing.

## License

MIT License
