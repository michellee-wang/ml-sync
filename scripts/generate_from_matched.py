"""
Generate Personalized MIDI from Matched Spotify Tracks

Uses pre-trained VAE model + matched MIDI files to generate music in user's style.

Pipeline:
1. Load matched MIDI files (from spotify_midi_matches.json)
2. Encode matched MIDIs to get user's latent distribution
3. Sample new latent vectors near user's style
4. Decode to generate personalized MIDI files

Usage:
    modal run scripts/generate_from_matched.py --num-samples 50 --temperature 0.8
"""

import modal
import sys
from pathlib import Path
from typing import List, Dict

# Create Modal app
app = modal.App("personalized-midi-generation")

# Volumes
models_volume = modal.Volume.from_name("trained-models")
dataset_volume = modal.Volume.from_name("lmd-dataset")
processed_volume = modal.Volume.from_name("lmd-processed")
output_volume = modal.Volume.from_name("generated-midi", create_if_missing=True)

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.2",
        "numpy==1.26.3",
        "pretty-midi==0.2.10",
        "tqdm",
    )
)

MODELS_PATH = "/models"
DATASET_PATH = "/data"
PROCESSED_PATH = "/processed"
OUTPUT_PATH = "/output"

@app.cls(
    image=image,
    gpu="T4",  # Smaller GPU for inference
    volumes={
        MODELS_PATH: models_volume,
        DATASET_PATH: dataset_volume,
        PROCESSED_PATH: processed_volume,
        OUTPUT_PATH: output_volume,
    },
    timeout=3600,
)
class PersonalizedGenerator:
    """Generate personalized MIDI based on user's Spotify taste"""

    def __init__(
        self,
        model_path: str = "pretrained_model.pt",
        matches_file: str = "spotify_midi_matches.json",
    ):
        """
        Initialize generator

        Args:
            model_path: Path to pre-trained model checkpoint
            matches_file: Path to spotify-MIDI matches JSON
        """
        import torch

        self.model_path = model_path
        self.matches_file = matches_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    def load_model(self, use_fp16: bool = True):
        """Load pre-trained VAE model with FP16 optimization"""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        # Inline MusicVAE definition (Modal API compatibility)
        class MusicVAE(nn.Module):
            def __init__(self, input_dim=88, latent_dim=512, hidden_dims=None, seq_length=512, beta=1.0):
                super().__init__()
                self.input_dim, self.latent_dim, self.seq_length, self.beta = input_dim, latent_dim, seq_length, beta

                self.encoder_conv = nn.Sequential(
                    nn.Conv2d(1, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                )

                self.conv_output_size = 256 * (input_dim // 16) * (seq_length // 16)
                self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
                self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
                self.decoder_fc = nn.Linear(latent_dim, self.conv_output_size)

                self.decoder_conv = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
                )

            def encode(self, x):
                h = self.encoder_conv(x).view(x.size(0), -1)
                return self.fc_mu(h), self.fc_logvar(h)

            def decode(self, z):
                h = self.decoder_fc(z).view(z.size(0), 256, self.input_dim // 16, self.seq_length // 16)
                return self.decoder_conv(h)

        print(f"\nLoading pre-trained VAE model...")

        checkpoint_path = Path(MODELS_PATH) / self.model_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint.get('config', {})

        # Create model
        model = MusicVAE(
            input_dim=88,
            latent_dim=config.get('latent_dim', 512),
            hidden_dims=[1024, 512, 256],
            seq_length=512,
            beta=config.get('beta', 0.1),
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)

        # OPTIMIZATION 2: Use FP16 for 2x speedup
        if use_fp16 and self.device.type == 'cuda':
            model = model.half()
            print("✓ Using FP16 (half precision) for 2x faster inference")

        model.eval()

        print("✓ Model loaded successfully")

        return model

    def load_matched_midis(self) -> List[Dict]:
        """Load matched MIDI file information"""
        import json

        matches_path = Path(PROCESSED_PATH) / self.matches_file

        if not matches_path.exists():
            raise FileNotFoundError(
                f"Matches file not found: {matches_path}\n"
                "Run: modal run scripts/match_spotify_to_midi.py first"
            )

        print(f"\nLoading matched MIDI files...")

        with open(matches_path, 'r') as f:
            data = json.load(f)

        matches = data.get('matched', [])

        if not matches:
            raise ValueError(
                "No matched tracks found! The user's Spotify tracks don't match any MIDI files.\n"
                "This could happen if the user listens to very new music not in the dataset."
            )

        print(f"✓ Found {len(matches)} matched tracks")

        # Extract MIDI file paths
        midi_info = []
        for match in matches:
            matched_track = match['matched_track']
            midi_files = matched_track['midi_files']
            spotify_track = match['spotify_track']

            for midi_file in midi_files:
                midi_info.append({
                    'path': midi_file,
                    'spotify_name': spotify_track['name'],
                    'spotify_artist': spotify_track['artist'],
                    'match_score': match['match_score'],
                })

        print(f"✓ Total MIDI files to load: {len(midi_info)}")

        return midi_info

    def midi_to_pianoroll(self, midi_path: Path) -> 'np.ndarray':
        """
        Load MIDI file and convert to piano roll

        Args:
            midi_path: Path to MIDI file

        Returns:
            Piano roll array (88, 512) or None if failed
        """
        import pretty_midi
        import numpy as np
        import random

        try:
            midi_data = pretty_midi.PrettyMIDI(str(midi_path))

            # Get piano roll (A0-C8, 88 keys)
            piano_roll = midi_data.get_piano_roll(fs=4)[21:109, :]

            # Normalize to [0, 1]
            piano_roll = np.clip(piano_roll / 127.0, 0, 1)

            # Extract 512-frame segment
            if piano_roll.shape[1] >= 512:
                start = random.randint(0, piano_roll.shape[1] - 512)
                piano_roll = piano_roll[:, start:start + 512]
            else:
                # Pad if too short
                piano_roll = np.pad(
                    piano_roll,
                    ((0, 0), (0, 512 - piano_roll.shape[1])),
                    mode='constant'
                )

            return piano_roll

        except Exception as e:
            print(f"  Warning: Failed to load {midi_path}: {e}")
            return None

    def get_cached_style_path(self):
        """Get path for cached user style"""
        return Path(PROCESSED_PATH) / "cached_user_style.pt"

    def load_cached_style(self, use_fp16: bool = True):
        """
        OPTIMIZATION 3: Load cached user style (instant!)

        Returns:
            (user_mu, user_std) tensors or None if no cache exists
        """
        import torch

        cache_path = self.get_cached_style_path()

        if not cache_path.exists():
            return None

        print(f"\n✓ Found cached user style! Loading (instant)...")

        cache = torch.load(cache_path, map_location=self.device)

        user_mu = cache['user_mu'].to(self.device)
        user_std = cache['user_std'].to(self.device)

        # Convert to FP16 if needed
        if use_fp16 and self.device.type == 'cuda':
            user_mu = user_mu.half()
            user_std = user_std.half()

        print(f"  ✓ Loaded cached style (skipped encoding {cache['num_encoded_files']} files)")
        print(f"  ✓ Saved ~30 seconds!")

        return user_mu, user_std

    def save_cached_style(self, user_mu, user_std, num_files: int):
        """
        OPTIMIZATION 3: Save user style to cache for instant future loads

        Args:
            user_mu: User's latent mean
            user_std: User's latent std
            num_files: Number of files encoded
        """
        import torch

        cache_path = self.get_cached_style_path()

        # Convert to CPU and FP32 for storage
        cache = {
            'user_mu': user_mu.cpu().float(),
            'user_std': user_std.cpu().float(),
            'num_encoded_files': num_files,
        }

        torch.save(cache, cache_path)
        processed_volume.commit()

        print(f"  ✓ Cached user style for future generations (instant reuse!)")

    def encode_user_style(self, model, midi_info: List[Dict], max_files: int = 100, use_fp16: bool = True):
        """
        Encode matched MIDI files to learn user's style distribution

        Args:
            model: Pre-trained VAE model
            midi_info: List of matched MIDI file info
            max_files: Maximum files to encode (for speed)
            use_fp16: Use FP16 precision

        Returns:
            mean and std of user's latent distribution
        """
        import torch
        import numpy as np
        from tqdm import tqdm

        print(f"\nEncoding user's style from matched MIDI files...")
        print(f"  Processing up to {max_files} files...")

        lmd_dir = Path(DATASET_PATH) / "lmd_full"
        latent_vectors = []

        # Load and encode MIDI files
        processed = 0
        for info in tqdm(midi_info[:max_files], desc="Encoding"):
            midi_path = lmd_dir / info['path']

            if not midi_path.exists():
                continue

            # Convert to piano roll
            piano_roll = self.midi_to_pianoroll(midi_path)

            if piano_roll is None:
                continue

            # Convert to torch tensor (add batch and channel dims)
            x = torch.FloatTensor(piano_roll[np.newaxis, np.newaxis, :, :]).to(self.device)

            # Convert to FP16 if model is FP16
            if use_fp16 and self.device.type == 'cuda':
                x = x.half()

            # Encode to latent space
            with torch.no_grad():
                mu, logvar = model.encode(x)
                # Use mean (mu) as the latent representation
                latent_vectors.append(mu.cpu().float().numpy())  # Always store as FP32

            processed += 1

        if not latent_vectors:
            raise ValueError("Failed to encode any MIDI files! Check the dataset.")

        # Stack all latent vectors
        latent_vectors = np.vstack(latent_vectors)

        # Compute user's style distribution
        user_mu = np.mean(latent_vectors, axis=0)
        user_std = np.std(latent_vectors, axis=0)

        # Convert to tensors
        user_mu = torch.FloatTensor(user_mu).to(self.device)
        user_std = torch.FloatTensor(user_std).to(self.device)

        # Convert to FP16 if needed
        if use_fp16 and self.device.type == 'cuda':
            user_mu = user_mu.half()
            user_std = user_std.half()

        print(f"\n✓ Encoded {processed} MIDI files")
        print(f"  User's latent distribution:")
        print(f"    Mean: {user_mu.float().mean():.4f} ± {user_mu.float().std():.4f}")
        print(f"    Std:  {user_std.float().mean():.4f} ± {user_std.float().std():.4f}")

        # OPTIMIZATION 3: Cache for future use
        self.save_cached_style(user_mu, user_std, processed)

        return user_mu, user_std

    def pianoroll_to_midi(
        self,
        piano_roll: 'np.ndarray',
        fs: int = 4,
        program: int = 0,
        threshold: float = 0.3,
        multi_instrument: bool = False,
    ) -> 'pretty_midi.PrettyMIDI':
        """
        Convert piano roll to MIDI file

        Args:
            piano_roll: Piano roll array (88, time)
            fs: Frames per second
            program: MIDI program (single instrument) or base program for multi
            threshold: Velocity threshold
            multi_instrument: If True, splits by pitch range into multiple instruments

        Returns:
            PrettyMIDI object
        """
        import pretty_midi
        import numpy as np

        midi = pretty_midi.PrettyMIDI()

        if multi_instrument:
            # Split into 3 instruments by pitch range
            # Low: Bass (0-29 keys) -> Bass or Cello
            # Mid: Harmony (30-59 keys) -> Piano or Guitar
            # High: Melody (60-87 keys) -> Lead instrument
            instruments = [
                pretty_midi.Instrument(program=33),  # Bass
                pretty_midi.Instrument(program=0 if program == 0 else program),  # Main instrument
                pretty_midi.Instrument(program=73 if program == 0 else min(program + 1, 127)),  # Lead/Melody
            ]
            pitch_ranges = [(0, 29), (30, 59), (60, 87)]
        else:
            # Single instrument for all notes
            instruments = [pretty_midi.Instrument(program=program)]
            pitch_ranges = [(0, 87)]

        # Threshold piano roll
        piano_roll_binary = (piano_roll > threshold).astype(int)

        # Find note on/off events
        padded_roll = np.pad(piano_roll_binary, ((0, 0), (1, 1)), mode='constant')
        diff = np.diff(padded_roll, axis=1)

        for inst_idx, (pitch_start, pitch_end) in enumerate(pitch_ranges):
            for pitch in range(pitch_start, pitch_end + 1):
                # Find note starts and ends
                starts = np.where(diff[pitch] == 1)[0]
                ends = np.where(diff[pitch] == -1)[0]

                for start, end in zip(starts, ends):
                    # Convert frame to time
                    start_time = start / fs
                    end_time = end / fs

                    # Get average velocity
                    velocity = int(piano_roll[pitch, start:end].mean() * 127)
                    velocity = max(1, min(127, velocity))

                    # Create note (pitch offset by 21 for A0)
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch + 21,
                        start=start_time,
                        end=end_time
                    )
                    instruments[inst_idx].notes.append(note)

        # Add instruments that have notes
        for inst in instruments:
            if len(inst.notes) > 0:
                midi.instruments.append(inst)

        return midi

    @modal.method()
    def generate(
        self,
        num_samples: int = 15,  # OPTIMIZATION 1: Reduced from 50 to 15 (faster!)
        temperature: float = 0.8,
        max_encode_files: int = 100,
        output_dir: str = "personalized",
        use_fp16: bool = True,  # OPTIMIZATION 2: FP16 for speed
        use_cache: bool = True,  # OPTIMIZATION 3: Cache for speed
        instrument: int = 0,  # MIDI instrument (0=Piano, 24=Guitar, 33=Bass, etc.)
        multi_instrument: bool = False,  # Use multiple instruments based on pitch ranges
    ):
        """
        Generate personalized MIDI files based on user's Spotify taste

        Args:
            num_samples: Number of samples to generate (default 15 for speed)
            temperature: Sampling temperature (controls variation around user's style)
            max_encode_files: Max matched files to encode for style learning
            output_dir: Output directory name
            use_fp16: Use FP16 precision (2x speedup on GPU)
            use_cache: Use cached user style if available (~30s speedup)
            instrument: MIDI instrument program number
                0 = Acoustic Grand Piano (default)
                24 = Acoustic Guitar (nylon)
                25 = Acoustic Guitar (steel)
                33 = Electric Bass (finger)
                40 = Violin
                48 = String Ensemble
                73 = Flute
                See: https://en.wikipedia.org/wiki/General_MIDI#Program_change_events
            multi_instrument: If True, intelligently splits notes into 3 instruments:
                - Low notes (0-29): Bass (MIDI 33)
                - Mid notes (30-59): Main instrument (specified by instrument param)
                - High notes (60-87): Lead/Melody (Flute or instrument+1)

        Returns:
            Dictionary with generated files and statistics
        """
        import torch
        import numpy as np
        from tqdm import tqdm

        # Instrument names for display
        instrument_names = {
            0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano",
            24: "Acoustic Guitar (nylon)", 25: "Acoustic Guitar (steel)",
            26: "Electric Guitar (jazz)", 27: "Electric Guitar (clean)",
            33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
            40: "Violin", 42: "Cello", 48: "String Ensemble",
            56: "Trumpet", 57: "Trombone", 73: "Flute", 74: "Recorder"
        }

        print("=" * 60)
        print("Personalized MIDI Generation (OPTIMIZED)")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Samples: {num_samples}")
        print(f"  Temperature: {temperature}")
        if multi_instrument:
            print(f"  Multi-Instrument Mode: YES")
            print(f"    Low (0-29): Bass (33)")
            print(f"    Mid (30-59): {instrument_names.get(instrument, 'Unknown')} ({instrument})")
            print(f"    High (60-87): Lead/Melody ({73 if instrument == 0 else min(instrument + 1, 127)})")
        else:
            print(f"  Instrument: {instrument} ({instrument_names.get(instrument, 'Unknown')})")
        print(f"  FP16: {use_fp16}")
        print(f"  Use Cache: {use_cache}")

        # OPTIMIZATION 2: Load model with FP16
        model = self.load_model(use_fp16=use_fp16)

        # OPTIMIZATION 3: Try to load cached style first
        user_mu, user_std = None, None
        midi_info = None

        if use_cache:
            cached = self.load_cached_style(use_fp16=use_fp16)
            if cached is not None:
                user_mu, user_std = cached

        # If no cache, encode user's style
        if user_mu is None:
            # Load matched MIDI info
            midi_info = self.load_matched_midis()

            # Encode user's style (this also saves cache)
            user_mu, user_std = self.encode_user_style(
                model, midi_info, max_encode_files, use_fp16=use_fp16
            )

        # Create output directory
        output_dir_path = Path(OUTPUT_PATH) / output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        generated_files = []

        print(f"\nGenerating {num_samples} personalized MIDI files...")
        print(f"  Sampling from user's style distribution...")

        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                # Sample from user's style distribution
                # z_new = mu + randn * std * temperature
                epsilon = torch.randn(1, model.latent_dim).to(self.device)
                z = user_mu.unsqueeze(0) + epsilon * user_std.unsqueeze(0) * temperature

                # Decode to piano roll
                output = model.decode(z, apply_sigmoid=True)

                # Remove batch and channel dimensions
                piano_roll = output[0, 0].cpu().numpy()

                # Convert to MIDI with selected instrument
                midi = self.pianoroll_to_midi(
                    piano_roll,
                    program=instrument,
                    threshold=0.3,
                    multi_instrument=multi_instrument
                )

                # Save MIDI file
                output_file = output_dir_path / f"personalized_{i:04d}.mid"
                midi.write(str(output_file))

                generated_files.append(str(output_file.relative_to(OUTPUT_PATH)))

        # Commit changes to volume
        output_volume.commit()

        print(f"\n✓ Generated {len(generated_files)} personalized MIDI files")
        print(f"  Saved to: {output_dir_path}")

        return {
            'generated_files': generated_files,
            'num_generated': len(generated_files),
            'num_matched_tracks': len(midi_info) if midi_info else 0,
            'temperature': temperature,
        }

    @modal.method()
    def generate_variations(
        self,
        num_variations: int = 10,
        temperature_range: tuple = (0.5, 1.5),
        max_encode_files: int = 100,
        output_dir: str = "variations",
    ):
        """
        Generate variations with different temperatures to show style range

        Args:
            num_variations: Number of variations per temperature
            temperature_range: (min, max) temperature range
            max_encode_files: Max matched files to encode
            output_dir: Output directory

        Returns:
            Dictionary with generated files
        """
        import torch
        import numpy as np
        from tqdm import tqdm

        print("=" * 60)
        print("Generating Style Variations")
        print("=" * 60)

        # Load model and encode user style
        model = self.load_model()
        midi_info = self.load_matched_midis()
        user_mu, user_std = self.encode_user_style(model, midi_info, max_encode_files)

        user_mu = torch.FloatTensor(user_mu).to(self.device)
        user_std = torch.FloatTensor(user_std).to(self.device)

        # Create output directory
        output_dir_path = Path(OUTPUT_PATH) / output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Generate variations at different temperatures
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_variations)
        generated_files = []

        print(f"\nGenerating {num_variations} variations from T={temperature_range[0]} to T={temperature_range[1]}...")

        with torch.no_grad():
            for i, temp in enumerate(tqdm(temperatures)):
                # Sample with this temperature
                epsilon = torch.randn(1, model.latent_dim).to(self.device)
                z = user_mu.unsqueeze(0) + epsilon * user_std.unsqueeze(0) * temp

                # Decode
                output = model.decode(z, apply_sigmoid=True)
                piano_roll = output[0, 0].cpu().numpy()

                # Convert to MIDI
                midi = self.pianoroll_to_midi(piano_roll, threshold=0.3)

                # Save with temperature in filename
                output_file = output_dir_path / f"variation_T{temp:.2f}_{i:04d}.mid"
                midi.write(str(output_file))

                generated_files.append(str(output_file.relative_to(OUTPUT_PATH)))

        output_volume.commit()

        print(f"\n✓ Generated {len(generated_files)} variations")

        return {
            'generated_files': generated_files,
            'temperatures': temperatures.tolist(),
        }

@app.function(
    image=image,
    volumes={OUTPUT_PATH: output_volume},
)
def list_generated_files():
    """List all generated MIDI files"""
    output_dir = Path(OUTPUT_PATH)

    if not output_dir.exists():
        return {"files": [], "total": 0}

    midi_files = list(output_dir.rglob("*.mid"))

    return {
        "files": [str(f.relative_to(output_dir)) for f in midi_files[:50]],
        "total": len(midi_files),
    }

@app.local_entrypoint()
def main(
    model_path: str = "pretrained_model.pt",
    matches_file: str = "spotify_midi_matches.json",
    num_samples: int = 15,  # OPTIMIZATION 1: Default 15 instead of 50
    temperature: float = 0.8,
    max_encode_files: int = 100,
    variations: bool = False,
    list_files: bool = False,
    use_fp16: bool = True,  # OPTIMIZATION 2: Enable FP16 by default
    use_cache: bool = True,  # OPTIMIZATION 3: Enable cache by default
    instrument: int = 0,  # MIDI instrument (0=Piano, 24=Guitar, etc.)
    multi_instrument: bool = False,  # Use multiple instruments based on pitch
):
    """
    Main entry point

    Args:
        model_path: Path to pre-trained model checkpoint
        matches_file: Path to spotify-MIDI matches JSON
        num_samples: Number of samples to generate
        temperature: Sampling temperature
        max_encode_files: Max matched files to encode for style
        variations: Generate temperature variations instead
        list_files: List generated files
    """
    if list_files:
        print("Listing generated files...")
        result = list_generated_files.remote()
        print(f"\n✓ Found {result['total']} generated MIDI files")
        print(f"\nFirst 50 files:")
        for f in result['files']:
            print(f"  {f}")
        return

    generator = PersonalizedGenerator(
        model_path=model_path,
        matches_file=matches_file,
    )

    if variations:
        print(f"Generating temperature variations...")
        result = generator.generate_variations.remote(
            num_variations=num_samples,
            temperature_range=(0.5, 1.5),
            max_encode_files=max_encode_files,
            output_dir="variations"
        )
    else:
        print(f"Generating {num_samples} personalized MIDI files (OPTIMIZED)...")
        result = generator.generate.remote(
            num_samples=num_samples,
            temperature=temperature,
            max_encode_files=max_encode_files,
            output_dir="personalized",
            use_fp16=use_fp16,
            use_cache=use_cache,
            instrument=instrument,
            multi_instrument=multi_instrument,
        )

    print(f"\n✓ Generation complete!")
    print(f"  Generated {result['num_generated']} files")
    print(f"\n  Saved to Modal volume 'generated-midi'")
    print(f"\n  To download: modal volume get generated-midi [local-path]")

if __name__ == "__main__":
    print("Generate personalized MIDI based on Spotify taste (OPTIMIZED):")
    print("  modal run scripts/generate_from_matched.py  # Generates 15 songs (fast!)")
    print("  modal run scripts/generate_from_matched.py --num-samples 20 --temperature 0.8")
    print("  modal run scripts/generate_from_matched.py --instrument 24  # Acoustic Guitar")
    print("  modal run scripts/generate_from_matched.py --instrument 73  # Flute")
    print("  modal run scripts/generate_from_matched.py --multi-instrument  # Bass + Main + Lead")
    print("  modal run scripts/generate_from_matched.py --instrument 24 --multi-instrument  # Multi-instrument with Guitar base")
    print("  modal run scripts/generate_from_matched.py --variations --num-samples 20")
    print("\nPopular instruments:")
    print("  0 = Piano (default), 24 = Guitar, 33 = Bass, 40 = Violin")
    print("  48 = Strings, 56 = Trumpet, 73 = Flute")
    print("  Full list: https://en.wikipedia.org/wiki/General_MIDI#Program_change_events")
    print("\nMulti-instrument mode:")
    print("  --multi-instrument splits notes by pitch into 3 instruments:")
    print("    Low (0-29): Bass (MIDI 33)")
    print("    Mid (30-59): Your chosen instrument (or Piano)")
    print("    High (60-87): Lead/Melody (Flute or next instrument)")
    print("\nOptimizations enabled by default:")
    print("  - FP16 inference (2x faster)")
    print("  - Cached user style (30s faster after first run)")
    print("  - 15 songs instead of 50 (3x faster)")
    print("\nTo list generated files:")
    print("  modal run scripts/generate_from_matched.py --list-files")
