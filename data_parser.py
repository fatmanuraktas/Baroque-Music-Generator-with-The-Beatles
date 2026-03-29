import os
import logging
import numpy as np
import pretty_midi

# Configure secure logging for parsing operations
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_vocal_melody(midi_path: str, sampling_rate: int = 4) -> np.ndarray:
    """
    Securely parses a MIDI file and extracts the vocal melody into a 1D numpy array.
    
    Args:
        midi_path (str): The file path to the Beatles MIDI file.
        sampling_rate (int): Number of samples per second (e.g., 4 captures 16th notes at 60 BPM).
        
    Returns:
        np.ndarray: A 1D array of float32 representing the vocal pitches over time.
    """
    # 1. Security: File existence and size validation (Defense against DoS/large files)
    if not os.path.exists(midi_path):
        logging.error(f"File not found: {midi_path}")
        raise FileNotFoundError(f"MIDI file missing at {midi_path}")
        
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB limit to prevent memory exhaustion
    if os.path.getsize(midi_path) > MAX_FILE_SIZE:
        logging.error(f"File exceeds maximum allowed size (5MB): {midi_path}")
        raise ValueError("MIDI file is too large. Potential security risk or memory leak.")

    try:
        # 2. Load MIDI data securely
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        
        # 3. Locate the target track 
        # For this pipeline, we assume the vocal melody is the first instrument track in the prepared MIDI
        if len(midi_data.instruments) == 0:
            logging.error("No instruments found in the MIDI file.")
            raise ValueError("Empty MIDI file provided.")
            
        vocal_track = midi_data.instruments[0] 
        logging.info(f"Extracting melody from track: {vocal_track.name if vocal_track.name else 'Track 0'}")
        
        # 4. Synthesize the piano roll to a 1D array
        # piano_roll shape: (128 pitches, time_steps)
        piano_roll = vocal_track.get_piano_roll(fs=sampling_rate)
        
        # Extract the highest pitch at each time step to create a monophonic vocal line
        melody_sequence = np.zeros(piano_roll.shape[1], dtype=np.float32)
        
        for step in range(piano_roll.shape[1]):
            active_pitches = np.nonzero(piano_roll[:, step])[0]
            if len(active_pitches) > 0:
                # Vocals are monophonic; take the highest note if multiple are playing
                melody_sequence[step] = float(np.max(active_pitches))
            else:
                # Rest (no note playing at this step)
                melody_sequence[step] = 0.0
                
        # 5. Robustness: Filter out trailing zeros (silence at the end of the track)
        # This prevents the RL agent from training on endless empty steps
        if np.any(melody_sequence):
            last_valid_index = np.max(np.nonzero(melody_sequence))
            melody_sequence = melody_sequence[:last_valid_index + 1]
        else:
            logging.warning("The extracted melody contains only silence.")

        logging.info(f"Successfully extracted vocal melody. Array length: {len(melody_sequence)} steps.")
        return melody_sequence

    except Exception as e:
        logging.error(f"Failed to parse MIDI file securely: {e}")
        raise