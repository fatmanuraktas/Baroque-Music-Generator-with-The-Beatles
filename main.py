import os
import logging
from data_parser import parse_vocal_melody
from agent import train_ppo_agent

# Configure secure logging for the main execution pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main execution pipeline for the Baroque-Beatles AI project.
    Handles data ingestion, validation, and initiates the RL training loop.
    """
    # Define paths securely using absolute paths to prevent directory traversal issues
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    midi_filename = "beatles_sample.mid"
    midi_path = os.path.join(data_dir, midi_filename)

    # Security: Ensure data directory exists, create if missing
    if not os.path.exists(data_dir):
        logging.warning(f"Data directory missing. Creating securely at: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
        logging.info(f"Please place your target MIDI file ('{midi_filename}') in the '{data_dir}' folder and rerun.")
        return

    # Check if the specific MIDI file exists before starting the heavy pipeline
    if not os.path.exists(midi_path):
        logging.error(f"Target MIDI file not found: {midi_path}")
        logging.info("Please add the MIDI file to proceed with the training pipeline.")
        return

    try:
        # Step 1: Parse the vocal melody securely
        logging.info("Step 1: Parsing Beatles vocal melody...")
        vocal_melody_array = parse_vocal_melody(midi_path=midi_path, sampling_rate=4)

        # Step 2: Validate the parsed data before passing it to the RL agent
        if len(vocal_melody_array) == 0:
            logging.error("Parsed melody array is empty. Aborting pipeline.")
            return

        # Step 3: Initiate the training sequence
        logging.info("Step 2: Initiating PPO Agent training sequence...")
        # Starting with 50,000 timesteps for the initial test run. 
        # This can be scaled up to 500,000+ for production training.
        train_ppo_agent(vocal_melody_data=vocal_melody_array, total_timesteps=50000)

        logging.info("Pipeline executed successfully. The Baroque Judge has completed its session.")

    except Exception as e:
        logging.error(f"Pipeline execution failed due to an unhandled exception: {e}")
        # In a production environment, this might trigger an alert to the security/ops team

if __name__ == "__main__":
    main()