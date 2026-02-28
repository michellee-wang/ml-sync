# Simple Guide - Spotify → MIDI Generation

## The Correct Pipeline

**ONE-TIME (Admin):**
1. Pre-train model on 176k MIDI files (~3 hours)

**PER-USER (Instant):**
1. Frontend sends Spotify tracks → API
2. Match tracks to MIDI files
3. Generate music using pre-trained model + matched tracks as style reference

**No per-user training needed!**

## Setup (One-Time)

### 1. Install Modal & Authenticate
```bash
pip install modal
modal token new
```

### 2. Download Dataset (~30 min, $0.10)
```bash
modal run scripts/modal_download_lmd.py
```

Downloads 176,581 MIDI files to Modal Volume.

### 3. Pre-train Model (~3 hours, $15)
```bash
modal run scripts/pretrain_model.py --gpu A100 --epochs 50
```

Creates a general-purpose music generation model. **Do this ONCE.**

## Per-User Flow

### 1. Frontend Sends Spotify Tracks

```typescript
// Frontend (you already have Spotify integration)
const topTracks = await getSpotifyTopTracks(); // Your existing code

// Send to API
await fetch('http://localhost:8000/api/ml/save-spotify-tracks', {
  method: 'POST',
  body: JSON.stringify({
    tracks: topTracks.map(t => ({
      name: t.name,
      artist: t.artists[0].name,
      spotify_id: t.id,
    }))
  })
});
```

### 2. Match Tracks to MIDIs (~1 min, $0.05)
```bash
modal run scripts/match_spotify_to_midi.py
```

Finds MIDI files matching the user's Spotify tracks.

### 3. Generate Music (~2 min, $0.10)
```bash
# Generate 50 songs in user's style
modal run scripts/generate_from_matched.py --num-samples 50
```

Uses:
- Pre-trained model (general music knowledge)
- Matched MIDIs (user's style reference)

### 4. Download Music
```bash
modal volume get generated-midi ./user_music
```

Now you have `.mid` files to use in your game!

## How It Works

### Pre-trained Model
- Trained on ALL 176k MIDI files
- Learns general music patterns
- Saved once, reused for all users

### Per-User Generation
- Load user's matched MIDIs
- Encode to latent space
- Sample nearby (with user's style)
- Decode to new MIDI
- **Takes 2 minutes, not 3 hours!**

## Cost Breakdown

**One-Time:**
- Download dataset: $0.10
- Pre-train model: $15
- **Total: $15.10**

**Per-User:**
- Match tracks: $0.05
- Generate 50 songs: $0.10
- **Total: $0.15 per user**

With $250 budget:
- One-time setup: $15
- Remaining: $235
- **Can serve 1,500+ users!**

## Files

```
scripts/
├── modal_download_lmd.py        # Download 176k MIDIs (one-time)
├── pretrain_model.py            # Pre-train model (one-time)
├── match_spotify_to_midi.py     # Match user's tracks
├── generate_from_matched.py     # Generate music (instant!)
└── run_pipeline.sh              # Run per-user flow

src/
├── models/midi_generator.py     # MusicVAE architecture
└── api/
    ├── main.py                  # FastAPI server
    └── routes.py                # API endpoints
```

## Example Timeline

**Day 1 (Setup):**
- 9:00 AM: Download dataset (30 min)
- 9:30 AM: Start pre-training (3 hours)
- 12:30 PM: Pre-training complete ✓

**Day 2+ (Users):**
- User logs in with Spotify
- Frontend sends tracks (instant)
- Match tracks (1 min)
- Generate 50 songs (2 min)
- **Total: 3 minutes per user!**

## Next Steps

1. ✅ Run one-time setup (pre-train model)
2. ✅ Frontend sends Spotify tracks to API
3. ✅ Run per-user pipeline
4. ✅ Use generated MIDIs in game

That's it! No per-user training, instant generation.
