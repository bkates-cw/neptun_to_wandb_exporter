"""
Simple W&B Rich Media Upload Test
==================================
Generates and uploads fake images, audio, video, tables to verify W&B works.

NEW: Tests logging media series in tables to avoid fork step offset issues!
- images/samples_table: 10 images in a table (steps 0-9 as column values)
- audio/notes_table: 3 audio clips in a table
- video/clips_table: 2 video clips in a table

Usage:
    export WANDB_API_KEY="your_key"
    python upload_media.py
"""

import wandb
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io
import struct
import tempfile
import os


def generate_image(width=100, height=100, color=None):
    """Generate a random colored image with shapes."""
    if color is None:
        color = tuple(np.random.randint(50, 200, 3))
    
    img = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(img)
    
    # Add some shapes
    for _ in range(3):
        x1, y1 = np.random.randint(0, width//2, 2)
        x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
        shape_color = tuple(np.random.randint(0, 255, 3))
        draw.ellipse([x1, y1, x2, y2], fill=shape_color)
    
    return img


def generate_audio_wav(duration=1.0, sample_rate=22050, frequency=440):
    """Generate a simple sine wave WAV file."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    num_samples = len(audio_int16)
    
    wav_buffer.write(b'RIFF')
    wav_buffer.write(struct.pack('<I', 36 + num_samples * 2))
    wav_buffer.write(b'WAVE')
    wav_buffer.write(b'fmt ')
    wav_buffer.write(struct.pack('<I', 16))
    wav_buffer.write(struct.pack('<H', 1))
    wav_buffer.write(struct.pack('<H', 1))
    wav_buffer.write(struct.pack('<I', sample_rate))
    wav_buffer.write(struct.pack('<I', sample_rate * 2))
    wav_buffer.write(struct.pack('<H', 2))
    wav_buffer.write(struct.pack('<H', 16))
    wav_buffer.write(b'data')
    wav_buffer.write(struct.pack('<I', num_samples * 2))
    wav_buffer.write(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer


def generate_video_frames(n_frames=30, size=(64, 64)):
    """Generate video frames as numpy arrays."""
    frames = []
    for i in range(n_frames):
        # Create frame with moving circle
        img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 240
        
        # Moving circle position
        cx = int(size[0] / 2 + 20 * np.sin(2 * np.pi * i / n_frames))
        cy = int(size[1] / 2 + 20 * np.cos(2 * np.pi * i / n_frames))
        
        # Draw circle (simple)
        for y in range(size[1]):
            for x in range(size[0]):
                if (x - cx)**2 + (y - cy)**2 < 100:
                    img[y, x] = [76, 175, 80]
        
        frames.append(img)
    
    return np.array(frames)


def main():
    print("🚀 Starting W&B Rich Media Upload Test")
    print("=" * 50)
    
    # Initialize W&B
    run = wandb.init(
        project="rich-media-test",
        name="test-upload",
        tags=["test", "rich-media"],
    )
    
    print(f"📊 Run URL: {run.url}")
    
    # 1. Upload Images
    print("\n🖼️  Uploading images...")
    
    # Single image
    img = generate_image(100, 100, (100, 150, 200))
    wandb.log({"images/single_image": wandb.Image(img, caption="Single test image")})
    
    # Image series (logged individually with steps)
    print("  - Logging image series individually...")
    for i in range(5):
        img = generate_image(64, 64)
        wandb.log({"images/series": wandb.Image(img, caption=f"Frame {i}")}, step=i)

    # Image series as TABLE (avoids fork offset issues!)
    print("  - Logging image series as table...")
    table = wandb.Table(columns=["step", "image"])
    for i in range(10):
        img = generate_image(64, 64)
        table.add_data(i, wandb.Image(img, caption=f"Sample {i}"))
    wandb.log({"images/samples_table": table})

    # Numpy array as image
    np_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    wandb.log({"images/numpy_image": wandb.Image(np_img, caption="From numpy array")})

    print("  ✓ Images uploaded (including table with 10 samples)")
    
    # 2. Upload Audio
    print("\n🔊 Uploading audio...")
    
    # Save WAV to temp file
    wav_buffer = generate_audio_wav(duration=2.0, frequency=440)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_buffer.getvalue())
        wav_path = f.name
    
    wandb.log({"audio/test_tone": wandb.Audio(wav_path, sample_rate=22050, caption="440Hz sine wave")})
    os.unlink(wav_path)
    
    # Audio from numpy array
    sample_rate = 22050
    t = np.linspace(0, 1, sample_rate, False)
    audio_array = 0.5 * np.sin(2 * np.pi * 880 * t)  # 880Hz
    wandb.log({"audio/numpy_audio": wandb.Audio(audio_array, sample_rate=sample_rate, caption="880Hz from numpy")})

    # Audio series as TABLE
    print("  - Logging audio series as table...")
    audio_table = wandb.Table(columns=["step", "audio"])
    temp_files = []  # Keep track of temp files to delete later
    for i in range(3):
        freq = 440 * (2 ** (i/12))  # Musical notes
        wav_buffer = generate_audio_wav(duration=0.5, frequency=freq)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_buffer.getvalue())
            temp_path = f.name
        audio_table.add_data(i, wandb.Audio(temp_path, sample_rate=22050, caption=f"Note {i}"))
        temp_files.append(temp_path)

    # Log the table FIRST, then clean up temp files
    wandb.log({"audio/notes_table": audio_table})
    for temp_path in temp_files:
        os.unlink(temp_path)

    print("  ✓ Audio uploaded (including table with 3 notes)")
    
    # 3. Upload Video
    print("\n🎬 Uploading video...")

    frames = generate_video_frames(n_frames=30, size=(64, 64))
    wandb.log({"video/animation": wandb.Video(frames, fps=10, caption="Moving circle")})

    # Video series as TABLE
    print("  - Logging video series as table...")
    video_table = wandb.Table(columns=["step", "video"])
    for i in range(2):
        frames = generate_video_frames(n_frames=15, size=(64, 64))
        video_table.add_data(i, wandb.Video(frames, fps=10, caption=f"Video {i}"))
    wandb.log({"video/clips_table": video_table})

    print("  ✓ Video uploaded (including table with 2 clips)")
    
    # 4. Upload Tables
    print("\n📋 Uploading tables...")
    
    df = pd.DataFrame({
        "experiment": [f"exp_{i}" for i in range(10)],
        "accuracy": np.random.uniform(0.8, 0.99, 10),
        "loss": np.random.uniform(0.01, 0.2, 10),
        "epochs": np.random.randint(10, 100, 10),
    })
    wandb.log({"tables/results": wandb.Table(dataframe=df)})
    
    print("  ✓ Tables uploaded")
    
    # 5. Upload HTML
    print("\n🌐 Uploading HTML...")
    
    html_content = """
    <div style="padding:20px; background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius:10px; color:white;">
        <h2>Test Report</h2>
        <p>This is a <strong>rich HTML</strong> content test.</p>
        <ul>
            <li>Accuracy: 95.6%</li>
            <li>Loss: 0.042</li>
        </ul>
    </div>
    """
    wandb.log({"html/report": wandb.Html(html_content)})
    
    print("  ✓ HTML uploaded")
    
    # 6. Upload Metrics
    print("\n📈 Uploading metrics...")
    
    for step in range(100):
        loss = 2.0 * np.exp(-step / 30) + np.random.normal(0, 0.02)
        acc = 0.95 * (1 - np.exp(-step / 25)) + np.random.normal(0, 0.01)
        wandb.log({
            "metrics/loss": loss,
            "metrics/accuracy": max(0, min(1, acc)),
        }, step=step)
    
    print("  ✓ Metrics uploaded")
    
    # 7. Upload matplotlib figure
    print("\n📊 Uploading matplotlib figure...")
    
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x), label='sin(x)')
        ax.plot(x, np.cos(x), label='cos(x)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Trigonometric Functions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        wandb.log({"plots/trig_functions": wandb.Image(fig)})
        plt.close(fig)
        
        print("  ✓ Matplotlib figure uploaded")
    except ImportError:
        print("  ⚠ Matplotlib not installed, skipping")
    
    # Finish
    run.finish()
    
    print("\n" + "=" * 50)
    print("✅ ALL UPLOADS COMPLETE!")
    print(f"🔗 View at: {run.url}")
    print("=" * 50)


if __name__ == "__main__":
    main()