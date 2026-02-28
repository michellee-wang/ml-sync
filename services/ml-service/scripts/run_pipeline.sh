#!/bin/bash
# Spotify → MIDI Generation Pipeline (No Training Per User!)

set -e

echo "========================================"
echo "Spotify → MIDI Generation Pipeline"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if pre-trained model exists
echo -e "${GREEN}Checking Pre-trained Model...${NC}"
echo ""
read -p "Is the model pre-trained already? (y/n): " pretrained

if [ "$pretrained" != "y" ]; then
    echo ""
    echo "⚠️  You need to pre-train the model ONCE (admin task)"
    echo ""
    echo "Run these commands ONCE:"
    echo "  1. modal run scripts/modal_download_lmd.py     (download dataset, 30 min)"
    echo "  2. modal run scripts/pretrain_model.py --epochs 50  (train model, 3 hrs)"
    echo ""
    echo "After that, you can generate music for any user instantly!"
    exit 1
fi

echo "✓ Pre-trained model ready"
echo ""
echo "========================================"

# Step 1: Spotify tracks (from frontend)
echo -e "${GREEN}Step 1: Spotify Tracks${NC}"
echo "Make sure frontend has saved Spotify tracks to data/spotify_tracks.json"
echo ""
read -p "Spotify tracks ready? (y/n): " ready
if [ "$ready" != "y" ]; then
    echo ""
    echo "Frontend should POST tracks to API endpoint:"
    echo "  POST /api/ml/save-spotify-tracks"
    echo ""
    echo "Or manually create data/spotify_tracks.json"
    exit 1
fi

echo ""
echo "========================================"

# Step 2: Match Spotify tracks to MIDI files
echo -e "${GREEN}Step 2: Match Spotify Tracks to MIDI Files${NC}"
echo "This finds MIDIs that match the user's Spotify tracks"
echo ""
read -p "Run matching? (y/n): " match
if [ "$match" = "y" ]; then
    modal run scripts/match_spotify_to_midi.py --spotify-file data/spotify_tracks.json
    echo -e "${GREEN}✓ Matching complete${NC}"
else
    echo "Cancelled"
    exit 1
fi

echo ""
echo "========================================"

# Step 3: Generate music (NO TRAINING!)
echo -e "${GREEN}Step 3: Generate Music!${NC}"
echo "Uses pre-trained model + matched tracks as style reference"
echo ""
read -p "How many songs to generate? (10-100): " num_songs
num_songs=${num_songs:-10}

echo ""
echo "Generating $num_songs songs..."
modal run scripts/generate_from_matched.py \
    --num-samples $num_songs \
    --use-matched true \
    --temperature 0.9

echo -e "${GREEN}✓ Generated $num_songs songs${NC}"

echo ""
echo "========================================"
echo -e "${GREEN}✓ Pipeline Complete!${NC}"
echo "========================================"
echo ""

# Step 4: Download generated music
echo -e "${GREEN}Step 4: Download Generated Music${NC}"
echo ""
read -p "Download music now? (y/n): " download
if [ "$download" = "y" ]; then
    # Get output directory
    read -p "Output directory (default: ./my_music): " output_dir
    output_dir=${output_dir:-./my_music}

    echo ""
    echo "Downloading to: $output_dir"
    echo ""

    # Execute download
    modal volume get generated-midi "$output_dir"

    # Validate download
    if [ -d "$output_dir" ] && [ "$(ls -A "$output_dir" 2>/dev/null)" ]; then
        echo ""
        echo -e "${GREEN}✓ Download successful!${NC}"
        echo "Music saved to: $output_dir"

        # Count files
        file_count=$(find "$output_dir" -type f -name "*.mid" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$file_count" -gt 0 ]; then
            echo "Found $file_count MIDI file(s)"
        fi
    else
        echo ""
        echo -e "${YELLOW}⚠️  Warning: Download directory is empty or doesn't exist${NC}"
        echo "Please check Modal volume status or try downloading manually:"
        echo "  modal volume get generated-midi ./my_music"
    fi
else
    echo ""
    echo "Skipping download. You can download later with:"
    echo "  modal volume get generated-midi ./my_music"
fi

echo ""
echo "========================================"
echo "Generate more music anytime (instant!):"
echo "  modal run scripts/generate_from_matched.py --num-samples 50"
