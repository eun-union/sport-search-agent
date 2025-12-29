import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict
import flyte
import flyte.io

# GPU environment for Whisper transcription
# Using Debian base with PyTorch, CUDA, and Whisper installed
gpu_env = flyte.TaskEnvironment(
    name="podcast_transcriber_gpu",
    resources=flyte.Resources(
        cpu=4,
        memory="16Gi",
        gpu=1,
    ),
    image=flyte.Image.from_debian_base().with_apt_packages(
        "ffmpeg",  # Required by Whisper for audio processing
    ).with_pip_packages(
        "torch",
        "torchaudio",
        "openai-whisper",
    ),
)

# CPU environment for RSS parsing and download
cpu_env = flyte.TaskEnvironment(
    name="podcast_transcriber_cpu",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    image=flyte.Image.from_debian_base().with_pip_packages(
        "feedparser",
        "requests",
    ),
    depends_on=[gpu_env],  # Driver calls GPU tasks, so declare dependency
)


@cpu_env.task()
async def fetch_rss_feed(rss_url: str) -> List[Dict[str, str]]:
    """Fetch and parse RSS feed to extract podcast episodes."""
    import feedparser

    print(f"Fetching RSS feed from: {rss_url}")
    feed = feedparser.parse(rss_url)

    episodes = []
    for entry in feed.entries:
        # Extract audio URL from enclosures
        audio_url = None
        if hasattr(entry, 'enclosures') and entry.enclosures:
            for enclosure in entry.enclosures:
                if 'audio' in enclosure.get('type', ''):
                    audio_url = enclosure.get('href') or enclosure.get('url')
                    break

        if audio_url:
            episode = {
                'title': entry.get('title', 'Unknown'),
                'audio_url': audio_url,
                'published': entry.get('published', ''),
            }
            episodes.append(episode)
            print(f"Found episode: {episode['title']}")

    print(f"Total episodes found: {len(episodes)}")
    return episodes


@cpu_env.task()
async def download_audio(audio_url: str, title: str) -> flyte.io.File:
    """Download audio file from URL."""
    import requests

    print(f"Downloading: {title}")
    print(f"URL: {audio_url}")

    response = requests.get(audio_url, stream=True)
    response.raise_for_status()

    # Save to temporary file
    suffix = '.mp3'  # default
    if 'content-type' in response.headers:
        content_type = response.headers['content-type']
        if 'mp3' in content_type:
            suffix = '.mp3'
        elif 'mp4' in content_type or 'm4a' in content_type:
            suffix = '.m4a'
        elif 'wav' in content_type:
            suffix = '.wav'

    with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        temp_path = f.name

    print(f"Downloaded to: {temp_path}")
    return await flyte.io.File.from_local(temp_path)


@gpu_env.task()
async def transcribe_audio(audio_file: flyte.io.File, title: str) -> Dict[str, str]:
    """Transcribe audio using Whisper model on GPU."""
    import whisper
    import torch

    print(f"Transcribing: {title}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")

    # Load Whisper model (using 'base' model for balance of speed/accuracy)
    # Options: tiny, base, small, medium, large
    model = whisper.load_model("base")

    # Download the audio file from S3 to local disk
    # Whisper/ffmpeg cannot read S3 paths directly
    audio_path = await audio_file.download()
    print(f"Processing audio file: {audio_path}")

    # Transcribe
    result = model.transcribe(audio_path)

    transcript = {
        'title': title,
        'text': result['text'],
        'language': result.get('language', 'unknown'),
    }

    print(f"Transcription complete for: {title}")
    print(f"Language detected: {transcript['language']}")
    print(f"Transcript length: {len(transcript['text'])} characters")

    return transcript


@cpu_env.task()
async def podcast_transcriber_driver(
    rss_url: str = "https://api.substack.com/feed/podcast/10845.rss",
    max_episodes: int = 3,
) -> List[Dict[str, str]]:
    """
    Main driver workflow that orchestrates podcast transcription.

    Args:
        rss_url: URL of the podcast RSS feed
        max_episodes: Maximum number of episodes to transcribe (default: 3 to avoid long runs)
    """
    print(f"Starting podcast transcription workflow")
    print(f"RSS URL: {rss_url}")
    print(f"Max episodes: {max_episodes}")

    # Step 1: Fetch RSS feed
    episodes = await fetch_rss_feed(rss_url)

    if not episodes:
        print("No episodes found in RSS feed")
        return []

    # Limit episodes
    episodes = episodes[:max_episodes]
    print(f"Processing {len(episodes)} episodes")

    # Step 2 & 3: Download and transcribe episodes in parallel
    transcripts = []

    with flyte.group("download-and-transcribe"):
        tasks = []
        for episode in episodes:
            async def process_episode(ep=episode):
                # Download audio
                audio_file = await download_audio(ep['audio_url'], ep['title'])
                # Transcribe
                transcript = await transcribe_audio(audio_file, ep['title'])
                return transcript

            tasks.append(process_episode())

        transcripts = await asyncio.gather(*tasks)

    print(f"Completed transcription of {len(transcripts)} episodes")
    return transcripts


if __name__ == "__main__":
    flyte.init_from_config()

    # Run with default RSS URL and limit to 3 episodes for testing
    run = flyte.run(podcast_transcriber_driver)

    print(f"\nWorkflow started!")
    print(f"Run name: {run.name}")
    print(f"Run URL: {run.url}")
    print(f"\nMonitor progress with:")
    print(f"  flyte get logs {run.name}")
    print(f"\nInspect results in Python:")
    print(f"""
import flyte
import asyncio

flyte.init_from_config()
run = flyte.remote.Run.get(name="{run.name}")

async def get_result():
    result = await run.result()
    for transcript in result:
        print(f"\\n{'='*80}")
        print(f"Title: {{transcript['title']}}")
        print(f"Language: {{transcript['language']}}")
        print(f"\\nTranscript:")
        print(transcript['text'][:500])  # First 500 chars

asyncio.run(get_result())
""")
