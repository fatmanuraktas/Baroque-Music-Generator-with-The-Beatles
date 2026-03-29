import numpy as np
import pretty_midi
from stable_baselines3 import PPO
from environment import BaroqueCounterpointEnv
import logging

logging.basicConfig(level=logging.INFO)

def generate_baroque_bass(model_path, midi_source_path, output_filename="baroque_beatles_output.mid", sampling_rate=4):
    # 1. Load the trained model securely
    model = PPO.load(model_path)
    
    # 2. Extract the same vocal melody used for training
    # For a real test, you can use a different Beatles song here
    from data_parser import parse_vocal_melody
    vocal_melody = parse_vocal_melody(midi_source_path)
    
    # 3. Initialize environment for inference
    env = BaroqueCounterpointEnv(vocal_melody=vocal_melody)
    obs, _ = env.reset()
    
    # 4. Agent starts composing
    generated_bass_line = []
    terminated = False
    
    logging.info("AI is composing the Baroque bass line...")
    while not terminated:
        # Predict the best next note based on the trained policy
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        generated_bass_line.append(info["actual_bass_pitch"])
        
    # 5. Export to MIDI using pretty_midi
    output_midi = pretty_midi.PrettyMIDI()
    
    # Add original vocal (Track 0)
    vocal_track = pretty_midi.Instrument(program=0) # Piano for vocal
    # Add AI Generated Bass (Track 1)
    bass_track = pretty_midi.Instrument(program=32) # Acoustic Bass
    
    # Reconstruct the notes using the same sampling rate as parsing/training.
    fs = sampling_rate
    for i, pitch in enumerate(vocal_melody):
        if pitch > 0:
            note = pretty_midi.Note(velocity=90, pitch=int(pitch), start=i/fs, end=(i+1)/fs)
            vocal_track.notes.append(note)
            
    for i, pitch in enumerate(generated_bass_line):
        if pitch > 0:
            note = pretty_midi.Note(velocity=100, pitch=int(pitch), start=i/fs, end=(i+1)/fs)
            bass_track.notes.append(note)
            
    output_midi.instruments.append(vocal_track)
    output_midi.instruments.append(bass_track)
    output_midi.write(output_filename)
    logging.info(f"Composition finished! Result saved as: {output_filename}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Baroque bass MIDI using a trained PPO model.")
    parser.add_argument("--model", default="./models/ppo_baroque_beatles", help="Path to PPO model (without .zip)")
    parser.add_argument("--source", default="./data/beatles_sample.mid", help="Path to source vocal MIDI file")
    parser.add_argument("--output", default="./baroque_beatles_output.mid", help="Output MIDI filename")
    parser.add_argument("--sampling_rate", type=int, default=4, help="Sampling rate used for violin note reconstruction")

    args = parser.parse_args()

    generate_baroque_bass(
        model_path=args.model,
        midi_source_path=args.source,
        output_filename=args.output,
        sampling_rate=args.sampling_rate
    )