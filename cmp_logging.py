"""
Neptune vs W&B Logging - Side by Side Comparison
================================================
This script shows how to log the same data types in both Neptune and W&B.
Each section logs identical data to both platforms so you can see the differences.

Usage:
    export NEPTUNE_API_TOKEN="your_token"
    export NEPTUNE_PROJECT="workspace/project"
    export WANDB_API_KEY="your_key"
    python compare_logging.py
"""

import os
import io
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import scipy.io.wavfile as wav

import neptune
from neptune.types import File
import wandb

# Track temp files to delete AFTER neptune stops
temp_files_to_cleanup = []


# =============================================================================
# INITIALIZATION
# =============================================================================

# Neptune
neptune_run = neptune.init_run(
    project=os.environ.get("NEPTUNE_PROJECT"),
    api_token=os.environ.get("NEPTUNE_API_TOKEN"),
    tags=["comparison-test", "migration"],
    name="side-by-side-test",
    description="Testing logging patterns",
)

# W&B
wandb_run = wandb.init(
    project="comparison-test",
    entity=os.environ.get("WANDB_ENTITY"),
    tags=["comparison-test", "migration"],
    name="side-by-side-test",
    notes="Testing logging patterns",
)


# =============================================================================
# CONFIG / PARAMETERS
# =============================================================================

config = {
    "model": {"architecture": "ResNet50", "num_layers": 50, "dropout": 0.2},
    "training": {"optimizer": "AdamW", "learning_rate": 0.001, "batch_size": 32},
    "data": {"dataset": "CIFAR-10", "augmentation": True},
}

# Neptune: assign to namespace
neptune_run["parameters"] = config

# W&B: pass to config object
wandb.config.update(config)


# =============================================================================
# SCALAR VALUES (Final Metrics)
# =============================================================================

# Neptune: direct assignment
neptune_run["evaluation/accuracy"] = 0.956
neptune_run["evaluation/loss"] = 0.042
neptune_run["evaluation/f1_score"] = 0.948

# W&B: use summary for final metrics
wandb.run.summary["evaluation/accuracy"] = 0.956
wandb.run.summary["evaluation/loss"] = 0.042
wandb.run.summary["evaluation/f1_score"] = 0.948


# =============================================================================
# TIME SERIES METRICS
# =============================================================================

for step in range(100):
    train_loss = 2.5 * np.exp(-step / 25) + np.random.normal(0, 0.03)
    val_loss = 2.7 * np.exp(-step / 30) + np.random.normal(0, 0.05)
    train_acc = 0.95 / (1 + np.exp(-0.1 * (step - 30)))
    val_acc = 0.92 / (1 + np.exp(-0.08 * (step - 35)))
    lr = 0.001 * (0.95 ** (step // 10))
    
    # Neptune: separate .append() calls for each metric
    neptune_run["train/loss"].append(train_loss, step=step)
    neptune_run["train/accuracy"].append(train_acc, step=step)
    neptune_run["validation/loss"].append(val_loss, step=step)
    neptune_run["validation/accuracy"].append(val_acc, step=step)
    neptune_run["train/learning_rate"].append(lr, step=step)
    
    # W&B: single log() call with all metrics
    wandb.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "validation/loss": val_loss,
        "validation/accuracy": val_acc,
        "train/learning_rate": lr,
    }, step=step)


# =============================================================================
# IMAGES - Single Image
# =============================================================================

img_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

# Neptune: File.as_image()
neptune_run["images/random_sample"].upload(File.as_image(img_array))

# W&B: wandb.Image()
wandb.log({"images/random_sample": wandb.Image(img_array, caption="Random sample")})


# =============================================================================
# IMAGES - From Matplotlib
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
ax.grid(True, alpha=0.3)

# Neptune: upload figure directly
neptune_run["plots/trig_functions"].upload(fig)

# W&B: wrap in wandb.Image()
wandb.log({"plots/trig_functions": wandb.Image(fig)})

plt.close(fig)


# =============================================================================
# IMAGES - Series (Multiple Images)
# =============================================================================

# Neptune: use .append() with File.as_image()
for i in range(10):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    neptune_run["images/samples"].append(
        File.as_image(img),
        name=f"sample_{i}",
        description=f"Generated sample {i}"
    )

# W&B: use a Table (better for browsing multiple images)
image_table = wandb.Table(columns=["step", "image"])
for i in range(10):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    image_table.add_data(i, wandb.Image(img, caption=f"Sample {i}"))
wandb.log({"images/samples_table": image_table})


# =============================================================================
# AUDIO
# =============================================================================

sample_rate = 22050
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration), False)
audio_array = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
wav.write(audio_file.name, sample_rate, audio_array)
audio_path = audio_file.name
temp_files_to_cleanup.append(audio_path)

# Neptune: upload from file path
neptune_run["audio/tone"].upload(audio_path)

# W&B: can use file path or numpy array directly
wandb.log({"audio/tone": wandb.Audio(audio_path, sample_rate=sample_rate, caption="440Hz sine wave")})


# =============================================================================
# VIDEO (Animated GIF)
# =============================================================================

frames = []
for i in range(30):
    img = Image.new('RGB', (64, 64), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    cx = int(32 + 20 * np.sin(2 * np.pi * i / 30))
    cy = int(32 + 20 * np.cos(2 * np.pi * i / 30))
    draw.ellipse([cx-8, cy-8, cx+8, cy+8], fill=(76, 175, 80))
    frames.append(img)

gif_file = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
frames[0].save(gif_file.name, format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=0)
gif_path = gif_file.name
temp_files_to_cleanup.append(gif_path)

# Neptune: upload from file path
neptune_run["video/animation"].upload(gif_path)

# W&B: can use file path or numpy array of frames
frames_array = np.array([np.array(f) for f in frames])
wandb.log({"video/animation": wandb.Video(frames_array, fps=10, format="gif")})


# =============================================================================
# TABLES / DATAFRAMES
# =============================================================================

df = pd.DataFrame({
    "experiment": [f"exp_{i}" for i in range(10)],
    "accuracy": np.random.uniform(0.85, 0.96, 10).round(4),
    "loss": np.random.uniform(0.05, 0.20, 10).round(4),
    "epochs": np.random.randint(50, 100, 10),
})

# Neptune: upload as HTML table
neptune_run["tables/results"].upload(File.as_html(df))

# W&B: use wandb.Table
wandb.log({"tables/results": wandb.Table(dataframe=df)})


# =============================================================================
# HTML CONTENT
# =============================================================================

html_content = """
<div style="padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:10px; color:white;">
    <h2>Experiment Report</h2>
    <p><strong>Accuracy:</strong> 95.6%</p>
    <p><strong>Loss:</strong> 0.042</p>
    <p><strong>Status:</strong> Completed</p>
</div>
"""

# Neptune: upload as HTML file
neptune_run["reports/summary"].upload(File.from_content(html_content.encode(), extension="html"))

# W&B: use wandb.Html
wandb.log({"reports/summary": wandb.Html(html_content)})


# =============================================================================
# TEXT / STRING VALUES
# =============================================================================

# Neptune: direct assignment for strings
neptune_run["info/description"] = "Testing migration patterns"
neptune_run["info/author"] = "ML Team"
neptune_run["info/version"] = "1.0.0"

# W&B: use config for metadata strings
wandb.config.description = "Testing migration patterns"
wandb.config.author = "ML Team"
wandb.config.version = "1.0.0"


# =============================================================================
# STRING SERIES (Logs)
# =============================================================================

log_messages = [
    "[INFO] Starting training...",
    "[INFO] Epoch 1/100 - loss: 2.45",
    "[INFO] Epoch 50/100 - loss: 0.32",
    "[INFO] Epoch 100/100 - loss: 0.08",
    "[INFO] Training complete!",
]

# Neptune: append each line
for msg in log_messages:
    neptune_run["logs/training"].append(msg)

# W&B: use a Table
log_table = wandb.Table(columns=["step", "message"])
for i, msg in enumerate(log_messages):
    log_table.add_data(i, msg)
wandb.log({"logs/training": log_table})


# =============================================================================
# ARTIFACTS / FILES
# =============================================================================

model_data = b"FAKE_MODEL_WEIGHTS_" + os.urandom(100)

model_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
model_file.write(model_data)
model_file.close()
model_path = model_file.name
temp_files_to_cleanup.append(model_path)

# Neptune: upload directly
neptune_run["artifacts/model"].upload(model_path)

# W&B: use Artifact
artifact = wandb.Artifact(name="model", type="model")
artifact.add_file(model_path, name="model.pt")
wandb.log_artifact(artifact)


# =============================================================================
# TAGS (Adding After Init)
# =============================================================================

# Neptune: use sys/tags
neptune_run["sys/tags"].add(["production", "v2.0"])

# W&B: modify run.tags tuple
wandb.run.tags = wandb.run.tags + ("production", "v2.0")


# =============================================================================
# CLEANUP
# =============================================================================

# Neptune (wait for async uploads to finish)
neptune_run.stop()

# W&B
wandb_run.finish()

# Now safe to delete temp files
for f in temp_files_to_cleanup:
    if os.path.exists(f):
        os.unlink(f)

print("Done!")