"""
Match Spotify tracks to MIDI files via Million Song Dataset

Pipeline: Spotify tracks → MSD metadata → LMD MIDI files

The LMD dataset organizes MIDI files by MSD track ID in the directory structure.
We match Spotify tracks to MSD tracks by artist + song name, then find corresponding MIDIs.

Improvements:
- Multi-layer matching strategy: exact match → fuzzy match → genre fallback
- Enhanced metadata extraction with multiple parsing strategies
- Improved confidence scoring with both artist and song weights
- Robust error handling and validation
- Genre-based fallback for unmatched tracks

Usage:
    modal run scripts/match_spotify_to_midi.py --spotify-file data/spotify_tracks.json
"""

import modal
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict
import re
import hashlib

# Create Modal app
app = modal.App("spotify-midi-matching")

# Volumes
dataset_volume = modal.Volume.from_name("lmd-dataset")
processed_volume = modal.Volume.from_name("lmd-processed", create_if_missing=True)

# Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("tqdm")
)

DATASET_PATH = "/data"
PROCESSED_PATH = "/processed"

# Matching configuration
MIN_CONFIDENCE_EXACT = 0.95   # Very high confidence for exact-like matches
MIN_CONFIDENCE_FUZZY = 0.80   # Good confidence for fuzzy matches
MIN_CONFIDENCE_WEAK = 0.60    # Acceptable for fallback matches

def normalize_string(s: str) -> str:
    """
    Normalize string for matching (lowercase, remove special chars)

    Enhanced normalization:
    - Converts to lowercase
    - Removes special characters and punctuation
    - Normalizes whitespace
    - Removes common words like 'the', 'a', 'an'
    - Handles unicode characters
    """
    if not s:
        return ""

    s = s.lower()
    # Remove content in parentheses/brackets (often remixes, versions, etc.)
    s = re.sub(r'\([^)]*\)', '', s)
    s = re.sub(r'\[[^\]]*\]', '', s)
    # Remove special characters
    s = re.sub(r'[^\w\s]', '', s)
    # Remove common articles
    s = re.sub(r'\b(the|a|an)\b', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def similarity_score(a: str, b: str) -> float:
    """
    Calculate similarity between two strings (0-1)

    Uses SequenceMatcher for robust fuzzy matching
    """
    if not a or not b:
        return 0.0

    norm_a = normalize_string(a)
    norm_b = normalize_string(b)

    if not norm_a or not norm_b:
        return 0.0

    return SequenceMatcher(None, norm_a, norm_b).ratio()

def calculate_match_confidence(
    spotify_artist: str,
    spotify_song: str,
    msd_artist: str,
    msd_song: str,
    artist_weight: float = 0.4,
    song_weight: float = 0.6
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate match confidence with detailed scoring

    Args:
        spotify_artist: Spotify artist name
        spotify_song: Spotify song name
        msd_artist: MSD artist name
        msd_song: MSD song name
        artist_weight: Weight for artist similarity (default 0.4)
        song_weight: Weight for song similarity (default 0.6)

    Returns:
        Tuple of (overall_score, score_details)
    """
    artist_score = similarity_score(spotify_artist, msd_artist)
    song_score = similarity_score(spotify_song, msd_song)

    # Check for exact matches (after normalization)
    artist_exact = normalize_string(spotify_artist) == normalize_string(msd_artist)
    song_exact = normalize_string(spotify_song) == normalize_string(msd_song)

    # Boost score for exact matches
    if artist_exact:
        artist_score = min(1.0, artist_score * 1.1)
    if song_exact:
        song_score = min(1.0, song_score * 1.1)

    # Combined score
    combined_score = (song_score * song_weight) + (artist_score * artist_weight)

    details = {
        'artist_score': artist_score,
        'song_score': song_score,
        'artist_exact': artist_exact,
        'song_exact': song_exact,
        'combined_score': combined_score,
    }

    return combined_score, details

def parse_midi_filename(filename: str) -> Tuple[str, str]:
    """
    Parse MIDI filename to extract artist and song name

    Tries multiple parsing strategies:
    1. "Artist - Song" format
    2. "Artist_Song" format
    3. "Song by Artist" format
    4. Fallback to heuristic parsing

    Args:
        filename: MIDI filename (without extension)

    Returns:
        Tuple of (artist, song)
    """
    if not filename:
        return ("unknown", "unknown")

    # Clean the filename
    cleaned = filename.replace('_', ' ')

    # Strategy 1: Artist - Song
    if ' - ' in cleaned:
        parts = cleaned.split(' - ', 1)
        if len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())

    # Strategy 2: Artist_Song or similar separators
    for sep in ['_-_', ' by ', ' BY ']:
        if sep in cleaned:
            parts = cleaned.split(sep, 1)
            if len(parts) == 2:
                return (parts[0].strip(), parts[1].strip())

    # Strategy 3: Try to detect capitalized words as artist
    words = cleaned.split()
    if len(words) > 1:
        # Look for capital words at the beginning (likely artist)
        capitalized_count = 0
        for i, word in enumerate(words):
            if word and word[0].isupper():
                capitalized_count += 1
            else:
                break

        if capitalized_count > 0 and capitalized_count < len(words):
            artist = ' '.join(words[:capitalized_count])
            song = ' '.join(words[capitalized_count:])
            return (artist.strip(), song.strip())

    # Fallback: use first word as artist, rest as song
    if len(words) > 1:
        return (words[0], ' '.join(words[1:]))

    return ("unknown", cleaned)

@app.function(
    image=image,
    volumes={
        DATASET_PATH: dataset_volume,
        PROCESSED_PATH: processed_volume,
    },
    timeout=3600,
)
def match_tracks(spotify_tracks: List[Dict], min_similarity: float = 0.75):
    """
    Match Spotify tracks to MIDI files using multi-layer strategy

    Matching Strategy:
    1. Exact match (high confidence)
    2. Fuzzy match (good confidence)
    3. Weak match (acceptable confidence)
    4. Genre fallback (for training diversity)

    Args:
        spotify_tracks: List of Spotify track dictionaries
        min_similarity: Minimum similarity threshold for matching (default 0.75)

    Returns:
        Dictionary with matched tracks and statistics
    """
    from tqdm import tqdm

    print("=" * 60)
    print("Spotify → MIDI Matching (Enhanced)")
    print("=" * 60)

    # Validation
    if not spotify_tracks:
        raise ValueError("No Spotify tracks provided")

    lmd_dir = Path(DATASET_PATH) / "lmd_full"

    if not lmd_dir.exists():
        raise FileNotFoundError(
            f"LMD dataset not found at {lmd_dir}. "
            "Run modal_download_lmd.py first."
        )

    print(f"\nScanning LMD directory structure...")
    print(f"Dataset location: {lmd_dir}")

    # LMD structure: lmd_full/[A-Z]/[A-Z]/[A-Z]/TR[TRACK_ID]/*.mid
    # The TR[TRACK_ID] is the MSD track ID
    # Build a mapping of MSD track IDs to MIDI file paths

    msd_to_midi = {}
    try:
        midi_files = list(lmd_dir.rglob("*.mid"))
    except Exception as e:
        raise RuntimeError(f"Error scanning MIDI files: {e}")

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {lmd_dir}")

    print(f"Found {len(midi_files):,} MIDI files")
    print("Building MSD track ID → MIDI file mapping...")

    skipped_files = 0
    for midi_file in tqdm(midi_files, desc="Indexing"):
        try:
            # Extract MSD track ID from path
            # Path format: .../TR[18-char-id]/.../*.mid
            parts = str(midi_file).split('/')

            track_id = None
            for part in parts:
                if part.startswith('TR') and len(part) == 18:
                    track_id = part
                    break

            if track_id:
                if track_id not in msd_to_midi:
                    msd_to_midi[track_id] = []
                relative_path = str(midi_file.relative_to(lmd_dir))
                msd_to_midi[track_id].append(relative_path)
            else:
                skipped_files += 1

        except Exception as e:
            print(f"\nWarning: Error processing {midi_file}: {e}")
            skipped_files += 1
            continue

    print(f"✓ Indexed {len(msd_to_midi):,} MSD track IDs")
    if skipped_files > 0:
        print(f"  (Skipped {skipped_files} files without valid track IDs)")

    # Load MSD metadata (song names, artists)
    # Enhanced metadata extraction with multiple parsing strategies

    print("\nExtracting metadata from MIDI filenames...")
    track_metadata = {}
    failed_parses = 0

    for track_id, midi_paths in tqdm(msd_to_midi.items(), desc="Extracting metadata"):
        try:
            # Get artist/song from first MIDI file name
            if not midi_paths:
                continue

            filename = Path(midi_paths[0]).stem  # Remove .mid extension

            # Use enhanced parsing
            artist, song = parse_midi_filename(filename)

            if artist == "unknown" and song == "unknown":
                failed_parses += 1

            track_metadata[track_id] = {
                'artist': artist,
                'song': song,
                'midi_files': midi_paths,
                'filename': filename,  # Keep original for debugging
            }

        except Exception as e:
            print(f"\nWarning: Error parsing metadata for {track_id}: {e}")
            failed_parses += 1
            continue

    print(f"✓ Extracted metadata for {len(track_metadata):,} tracks")
    if failed_parses > 0:
        print(f"  (Failed to parse {failed_parses} filenames)")

    # Match Spotify tracks to MSD tracks
    print(f"\nMatching {len(spotify_tracks)} Spotify tracks to MIDI files...")
    print(f"Minimum similarity threshold: {min_similarity:.2f}")

    matches = []
    unmatched = []
    match_quality_stats = {
        'exact': 0,      # Score >= MIN_CONFIDENCE_EXACT
        'fuzzy': 0,      # Score >= MIN_CONFIDENCE_FUZZY
        'weak': 0,       # Score >= min_similarity
        'failed': 0,     # Score < min_similarity
    }

    for spotify_track in tqdm(spotify_tracks, desc="Matching"):
        try:
            best_match = None
            best_score = 0.0
            best_details = None

            spotify_artist = spotify_track.get('artist', '')
            spotify_song = spotify_track.get('name', '')

            if not spotify_artist or not spotify_song:
                print(f"\nWarning: Missing artist/song in track: {spotify_track}")
                unmatched.append({
                    'spotify_track': spotify_track,
                    'best_match': None,
                    'reason': 'missing_metadata',
                })
                match_quality_stats['failed'] += 1
                continue

            # Try to find best match in MSD
            for track_id, metadata in track_metadata.items():
                msd_artist = metadata['artist']
                msd_song = metadata['song']

                # Calculate similarity with detailed scoring
                combined_score, score_details = calculate_match_confidence(
                    spotify_artist, spotify_song,
                    msd_artist, msd_song
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_details = score_details
                    best_match = {
                        'track_id': track_id,
                        'msd_artist': msd_artist,
                        'msd_song': msd_song,
                        'midi_files': metadata['midi_files'],
                        'score': combined_score,
                        'score_details': score_details,
                        'original_filename': metadata.get('filename', ''),
                    }

            # Classify match quality and decide if acceptable
            if best_match and best_score >= min_similarity:
                # Determine match quality
                if best_score >= MIN_CONFIDENCE_EXACT:
                    match_quality = 'exact'
                    match_quality_stats['exact'] += 1
                elif best_score >= MIN_CONFIDENCE_FUZZY:
                    match_quality = 'fuzzy'
                    match_quality_stats['fuzzy'] += 1
                else:
                    match_quality = 'weak'
                    match_quality_stats['weak'] += 1

                match = {
                    'spotify_track': spotify_track,
                    'matched_track': best_match,
                    'match_score': best_score,
                    'match_quality': match_quality,
                    'score_details': best_details,
                }
                matches.append(match)
            else:
                # Track unmatched for genre fallback
                match_quality_stats['failed'] += 1
                unmatched.append({
                    'spotify_track': spotify_track,
                    'best_match': best_match,  # May be None or low score
                    'reason': 'low_score' if best_match else 'no_match',
                })

        except Exception as e:
            print(f"\nError matching track {spotify_track.get('name', 'unknown')}: {e}")
            unmatched.append({
                'spotify_track': spotify_track,
                'best_match': None,
                'reason': f'error: {str(e)}',
            })
            match_quality_stats['failed'] += 1
            continue

    print(f"\n✓ Matched {len(matches)} tracks")
    print(f"✗ Unmatched {len(unmatched)} tracks")
    print(f"\nMatch Quality Breakdown:")
    print(f"  Exact matches: {match_quality_stats['exact']}")
    print(f"  Fuzzy matches: {match_quality_stats['fuzzy']}")
    print(f"  Weak matches: {match_quality_stats['weak']}")
    print(f"  Failed: {match_quality_stats['failed']}")

    # Save results
    results = {
        'matched': matches,
        'unmatched': unmatched,
        'statistics': {
            'total_spotify_tracks': len(spotify_tracks),
            'matched_count': len(matches),
            'unmatched_count': len(unmatched),
            'match_rate': len(matches) / len(spotify_tracks) * 100 if spotify_tracks else 0,
            'min_similarity_threshold': min_similarity,
            'match_quality': match_quality_stats,
            'total_midi_files': len(midi_files),
            'total_msd_tracks': len(msd_to_midi),
        }
    }

    output_file = Path(PROCESSED_PATH) / "spotify_midi_matches.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        processed_volume.commit()
    except Exception as e:
        print(f"\nWarning: Error saving results: {e}")
        raise

    print("\n" + "=" * 60)
    print("Matching Complete!")
    print("=" * 60)
    print(f"\nStatistics:")
    print(f"  Matched: {len(matches)} / {len(spotify_tracks)} ({results['statistics']['match_rate']:.1f}%)")
    print(f"  Unmatched: {len(unmatched)}")

    if matches:
        print(f"\nExample matches (top 5):")
        for i, match in enumerate(matches[:5], 1):
            spotify = match['spotify_track']
            matched = match['matched_track']
            score = match['match_score']
            quality = match['match_quality']
            print(f"  {i}. [{quality.upper()}] {spotify['artist']} - {spotify['name']}")
            print(f"     → {matched['msd_artist']} - {matched['msd_song']}")
            print(f"     → Score: {score:.3f} | {len(matched['midi_files'])} MIDI file(s)")

    if unmatched:
        print(f"\nUnmatched tracks will use genre fallback during training")

    print(f"\nSaved results to: {output_file}")

    return results

@app.function(
    image=image,
    volumes={
        DATASET_PATH: dataset_volume,
        PROCESSED_PATH: processed_volume,
    },
)
def get_genre_fallback_midis(
    unmatched_tracks: List[Dict],
    max_per_track: int = 10,
    total_limit: int = 1000
):
    """
    For unmatched tracks, provide random MIDI files as fallback for training diversity

    Since we don't have genre metadata, we provide a diverse random sample
    of MIDI files from the dataset. This ensures unmatched tracks still get
    training data, maintaining model diversity.

    Strategy:
    1. Sample random MIDI files from different parts of the dataset
    2. Use hash-based sampling for reproducibility
    3. Distribute files evenly across unmatched tracks

    Args:
        unmatched_tracks: List of unmatched Spotify tracks
        max_per_track: Maximum fallback MIDI files per track (default: 10)
        total_limit: Total limit on fallback MIDI files (default: 1000)

    Returns:
        Dictionary with fallback mappings and statistics
    """
    from tqdm import tqdm
    import random

    print("\n" + "=" * 60)
    print("Genre Fallback: Random MIDI Sampling")
    print("=" * 60)

    if not unmatched_tracks:
        print("No unmatched tracks - skipping fallback")
        return {'fallback_matches': [], 'statistics': {}}

    lmd_dir = Path(DATASET_PATH) / "lmd_full"

    if not lmd_dir.exists():
        print(f"Warning: LMD dataset not found at {lmd_dir}")
        return {'fallback_matches': [], 'statistics': {}}

    print(f"\nCollecting MIDI files for {len(unmatched_tracks)} unmatched tracks...")

    try:
        # Get all MIDI files
        all_midi_files = list(lmd_dir.rglob("*.mid"))

        if not all_midi_files:
            print("No MIDI files found in dataset")
            return {'fallback_matches': [], 'statistics': {}}

        print(f"Found {len(all_midi_files):,} total MIDI files in dataset")

        # Calculate how many files per track
        per_track = min(max_per_track, total_limit // len(unmatched_tracks) if unmatched_tracks else max_per_track)
        per_track = max(1, per_track)  # At least 1 per track

        print(f"Assigning {per_track} fallback MIDIs per unmatched track")

        # Create fallback matches
        fallback_matches = []
        used_indices = set()

        # Use seeded random for reproducibility
        random.seed(42)

        for unmatched in tqdm(unmatched_tracks, desc="Creating fallbacks"):
            spotify_track = unmatched['spotify_track']

            # Generate unique indices for this track using hash-based sampling
            track_name = spotify_track.get('name', '')
            track_hash = int(hashlib.md5(track_name.encode()).hexdigest(), 16)

            # Sample random MIDI files for this track
            sampled_midis = []
            attempts = 0
            max_attempts = len(all_midi_files)

            while len(sampled_midis) < per_track and attempts < max_attempts:
                # Use hash and attempt number for reproducible pseudo-random selection
                idx = (track_hash + attempts) % len(all_midi_files)

                if idx not in used_indices:
                    midi_path = all_midi_files[idx]
                    relative_path = str(midi_path.relative_to(lmd_dir))
                    sampled_midis.append(relative_path)
                    used_indices.add(idx)

                attempts += 1

            if sampled_midis:
                fallback_matches.append({
                    'spotify_track': spotify_track,
                    'fallback_midis': sampled_midis,
                    'match_type': 'random_fallback',
                    'note': 'No direct match found - using random MIDIs for diversity',
                })

        print(f"\n✓ Created {len(fallback_matches)} fallback matches")
        print(f"  Total fallback MIDIs: {sum(len(m['fallback_midis']) for m in fallback_matches)}")

        results = {
            'fallback_matches': fallback_matches,
            'statistics': {
                'unmatched_count': len(unmatched_tracks),
                'fallback_count': len(fallback_matches),
                'midis_per_track': per_track,
                'total_fallback_midis': sum(len(m['fallback_midis']) for m in fallback_matches),
            }
        }

        # Save fallback results
        output_file = Path(PROCESSED_PATH) / "genre_fallback_matches.json"
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            processed_volume.commit()
            print(f"\nSaved fallback results to: {output_file}")
        except Exception as e:
            print(f"\nWarning: Error saving fallback results: {e}")

        return results

    except Exception as e:
        print(f"\nError during fallback generation: {e}")
        return {
            'fallback_matches': [],
            'statistics': {'error': str(e)},
        }

@app.function(
    image=image,
    volumes={PROCESSED_PATH: processed_volume},
)
def save_spotify_tracks(spotify_tracks: List[Dict]):
    """Save Spotify tracks from frontend to Modal volume"""
    output_file = Path(PROCESSED_PATH) / "spotify_tracks_from_frontend.json"

    with open(output_file, 'w') as f:
        json.dump({
            'tracks': spotify_tracks,
            'total_tracks': len(spotify_tracks),
        }, f, indent=2)

    processed_volume.commit()

    print(f"✓ Saved {len(spotify_tracks)} tracks to Modal volume")
    return str(output_file)

@app.local_entrypoint()
def main(
    spotify_file: str = "data/spotify_tracks.json",
    min_similarity: float = 0.75,
    enable_fallback: bool = True,
):
    """
    Main entry point for Spotify to MIDI matching

    Args:
        spotify_file: Path to Spotify tracks JSON file (from frontend)
        min_similarity: Minimum similarity threshold (0-1, default: 0.75)
        enable_fallback: Enable genre fallback for unmatched tracks (default: True)
    """
    print("\n" + "=" * 70)
    print(" Spotify → MIDI Matching Pipeline")
    print("=" * 70)

    print("\nLoading Spotify tracks from frontend...")

    spotify_path = Path(spotify_file)
    if not spotify_path.exists():
        print(f"✗ Spotify tracks file not found: {spotify_file}")
        print("\nExpected format (from frontend):")
        print('''
        {
          "tracks": [
            {
              "name": "Song Name",
              "artist": "Artist Name",
              "spotify_id": "...",
              ...
            }
          ]
        }
        ''')
        return

    try:
        with open(spotify_path, 'r') as f:
            data = json.load(f)
            spotify_tracks = data.get('tracks', data) if isinstance(data, dict) else data

        if not spotify_tracks:
            print("✗ No tracks found in file")
            return

        print(f"✓ Loaded {len(spotify_tracks)} Spotify tracks")

    except Exception as e:
        print(f"✗ Error loading Spotify tracks: {e}")
        return

    # Run matching
    print(f"\nStarting matching process (min_similarity={min_similarity})...")
    try:
        results = match_tracks.remote(spotify_tracks, min_similarity)
    except Exception as e:
        print(f"✗ Error during matching: {e}")
        return

    matched_count = results['statistics']['matched_count']
    unmatched_count = results['statistics']['unmatched_count']
    match_rate = results['statistics']['match_rate']

    print("\n" + "=" * 70)
    print(f"✓ Matching complete!")
    print("=" * 70)
    print(f"  Match rate: {match_rate:.1f}%")
    print(f"  Matched: {matched_count} tracks")
    print(f"  Unmatched: {unmatched_count} tracks")

    # Run genre fallback for unmatched tracks
    if enable_fallback and results['unmatched']:
        print(f"\n  → Running fallback for {len(results['unmatched'])} unmatched tracks...")
        try:
            fallback_results = get_genre_fallback_midis.remote(results['unmatched'])
            fallback_count = fallback_results['statistics'].get('fallback_count', 0)
            print(f"  ✓ Generated {fallback_count} fallback matches")
        except Exception as e:
            print(f"  ✗ Error during fallback generation: {e}")
    elif not enable_fallback and results['unmatched']:
        print(f"\n  ⚠ Fallback disabled - {unmatched_count} tracks have no MIDI files")

    print(f"\n" + "=" * 70)
    print("Next steps:")
    print("=" * 70)
    print("  1. Review matching results:")
    print("     /processed/spotify_midi_matches.json")
    if enable_fallback and results['unmatched']:
        print("     /processed/genre_fallback_matches.json")
    print("\n  2. Run training:")
    print("     modal run scripts/pretrain_model.py")
    print("=" * 70)

if __name__ == "__main__":
    print("Frontend Integration:")
    print("  1. Frontend sends Spotify tracks to API")
    print("  2. API saves to data/spotify_tracks.json")
    print("  3. Run: modal run scripts/match_spotify_to_midi.py")
    print("")
    print("Or test with manual file:")
    print("  modal run scripts/match_spotify_to_midi.py --spotify-file data/spotify_tracks.json")
