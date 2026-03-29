import os
import numpy as np
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the environment from the separate file
from environment import BaroqueCounterpointEnv

# Configure secure logging for training audit and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_ppo_agent(vocal_melody_data: np.ndarray, total_timesteps: int = 100000):
    """
    Initializes, trains, and securely saves the PPO agent for Baroque counterpoint generation.
    """
    model_save_path = "./models/ppo_baroque_beatles"
    log_dir = "./logs/"
    
    # Security/Robustness: Ensure directories exist before writing files to prevent IO errors
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    try:
        # 1. Initialize the custom environment
        logging.info("Initializing BaroqueCounterpointEnv...")
        env = BaroqueCounterpointEnv(vocal_melody=vocal_melody_data)
        
        # 2. Security Check: Validate the custom environment complies with Stable Baselines3 API
        logging.info("Running environment API check...")
        check_env(env, warn=True)
        
        # 3. Setup Checkpoint Callback (Defense against sudden training interruptions)
        # Saves the model every 10,000 steps
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=log_dir, 
            name_prefix="ppo_baroque_checkpoint"
        )

        # 4. Initialize PPO Model
        # MlpPolicy is used since our observation space is a 1D vector.
        # tensorboard_log allows for monitoring training metrics securely offline.
        logging.info("Initializing PPO model architecture...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log=log_dir,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64
        )

        # 5. Train the Agent
        logging.info(f"Starting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps, 
            callback=checkpoint_callback,
            tb_log_name="PPO_Baroque_Run"
        )

        # 6. Securely save the final model
        model.save(model_save_path)
        logging.info(f"Final model successfully saved to {model_save_path}.zip")

    except Exception as e:
        logging.error(f"A critical error occurred during agent training: {e}")
        raise

if __name__ == "__main__":
    # Mock data for initial pipeline testing (e.g., a simple scale imitating a melody)
    # In production, this will be replaced with parsed MIDI data from early Beatles songs
    mock_vocal_data = np.array([60, 62, 64, 65, 67, 69, 71, 72] * 4, dtype=np.float32)
    
    logging.info("Starting agent training script...")
    train_ppo_agent(mock_vocal_data, total_timesteps=50000)