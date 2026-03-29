import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

# Set up secure and trackable logging for debugging and audit trails
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaroqueCounterpointEnv(gym.Env):
    """
    Custom environment for generating Baroque bass lines (Basso Continuo) 
    against a static Beatles vocal melody using PPO.
    """
    def __init__(self, vocal_melody: np.ndarray, context_length: int = 4):
        super(BaroqueCounterpointEnv, self).__init__()
        
        # Security & Stability: Strict input validation to prevent runtime crashes
        if not isinstance(vocal_melody, np.ndarray) or len(vocal_melody) == 0:
            logging.error("Invalid vocal_melody array provided.")
            raise ValueError("vocal_melody must be a non-empty numpy array.")
            
        self.vocal_melody = vocal_melody
        self.melody_length = len(vocal_melody)
        
        # How many previous bass notes the agent remembers
        self.context_length = context_length 
        self.current_step = 0
        
        # Action Space: MIDI notes. Restrict to human-audible frequency range.
        # Human hearing ~20Hz to 20kHz. MIDI note 21 is A0 (27.5Hz), note 127 is G9 (~12543Hz).
        # (full 20kHz would be ~MIDI 135, outside standard 0-127), so we clamp to the standard MIDI range
        # while respecting the human hearing lower bound.
        self.min_bass_pitch = 21   # A0, lowest practical audible keyboard note
        self.max_bass_pitch = 127  # highest standard MIDI note in audible range for this project
        self.action_space = spaces.Discrete(self.max_bass_pitch - self.min_bass_pitch + 1)

        # Observation Space: [Current Vocal Pitch, Rhythmic Beat, Last N Bass Pitches]
        # We use float32 to ensure compatibility with stable-baselines3 algorithms
        obs_dim = 2 + self.context_length
        self.observation_space = spaces.Box(
            low=0.0, 
            high=127.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        self.bass_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Initialize history with rests (0.0)
        self.bass_history = [0.0] * self.context_length
        
        return self._get_observation(), {}

    def step(self, action: int):
        # 1. Defense in Depth: Validate action against the action space
        if not self.action_space.contains(action):
            logging.warning(f"Invalid action received from agent: {action}. Defaulting to 0.")
            action = 0 
            
        actual_pitch = action + self.min_bass_pitch
        
        # 2. Update internal state and memory
        self.bass_history.append(float(actual_pitch))
        if len(self.bass_history) > self.context_length:
            self.bass_history.pop(0)
            
        # 3. Evaluate the action using the Reward Function (The Baroque Judge)
        reward = self._calculate_reward(actual_pitch)
        
        # 4. Advance step and check terminal state
        self.current_step += 1
        terminated = bool(self.current_step >= self.melody_length)
        truncated = False 
        
        # Prevent index out of bounds on the final step by returning a zeroed observation
        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()
            
        info = {"actual_bass_pitch": actual_pitch}
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.float32:
        current_vocal = float(self.vocal_melody[self.current_step])
        # Simple rhythmic context (e.g., measuring beat position in a 4/4 time signature)
        beat_position = float(self.current_step % 4) 
        
        obs = [current_vocal, beat_position] + self.bass_history
        return np.array(obs, dtype=np.float32)

    
    def _calculate_reward(self, current_bass_pitch: int) -> float:
        """
        Evaluates the generated bass note against strict Baroque counterpoint rules.
        Includes safety checks for array indexing based on previous debugs.
        """
        # Security: Prevent out-of-bounds indexing if step exceeds melody length
        if self.current_step >= self.melody_length:
            logging.warning("Reward calculation attempted out of bounds. Returning 0.0")
            return 0.0

        reward = 0.0
        current_vocal = float(self.vocal_melody[self.current_step])
        
        # 1. VERTICAL HARMONY (Consonance vs. Dissonance)
        # Calculate the interval modulo 12 (ignoring octaves for basic harmony check)
        interval = abs(current_vocal - current_bass_pitch) % 12
        
        # Consonances: Unison(0), Minor 3rd(3), Major 3rd(4), Perfect 5th(7), Minor/Major 6th(8,9)
        consonances = [0, 3, 4, 7, 8, 9]
        if interval in consonances:
            reward += 1.0
        else:
            # Penalize dissonances (2nds, 4ths, 7ths) heavily in first-species counterpoint
            reward -= 1.5 
            
        # 2. HORIZONTAL VOICE LEADING & MOTION (Requires analyzing t and t-1)
        # Ensure we have a valid previous step and history to compare against
        if self.current_step > 0 and len(self.bass_history) > 0:
            
            prev_vocal = float(self.vocal_melody[self.current_step - 1])
            # Retrieve the last valid bass note before the current action was taken
            prev_bass = self.bass_history[-1] 
            
            vocal_diff = current_vocal - prev_vocal
            bass_diff = current_bass_pitch - prev_bass
            
            # A. Melodic Constraints for Bass Line
            leap_size = abs(bass_diff)
            if leap_size > 12: 
                # Leaping more than an octave is highly uncharacteristic for Baroque bass
                reward -= 2.0
            elif leap_size == 6: 
                # Tritone leap (The Devil in Music - Diabolus in Musica) is strictly forbidden
                reward -= 3.0
                
            # B. Motion Rules Between Voices (The core of Counterpoint)
            # Check if voices are moving in opposite directions (Contrary Motion)
            if vocal_diff * bass_diff < 0:
                reward += 2.0  # Holy grail of counterpoint, highly rewarded
                
            # Check if voices are moving in the same direction (Similar/Parallel Motion)
            elif vocal_diff * bass_diff > 0:
                prev_interval = abs(prev_vocal - prev_bass) % 12
                
                # Parallel Fifths Check
                if prev_interval == 7 and interval == 7:
                    reward -= 10.0  # Fatal error in strict Baroque theory
                    
                # Parallel Octaves/Unisons Check
                elif prev_interval == 0 and interval == 0:
                    reward -= 10.0  # Fatal error
                    
                # Hidden/Direct 5ths or Octaves
                # (Approaching a perfect consonance by similar motion)
                elif interval in [0, 7]:
                    reward -= 2.0
                    
            # Check for Oblique Motion (One voice stays the same, the other moves)
            elif vocal_diff == 0 and bass_diff != 0:
                reward += 1.0 # Acceptable and good for variety
            elif bass_diff == 0 and vocal_diff != 0:
                reward += 1.0 # Acceptable

        return float(reward)