"""
Training script for micro-rnn
==============================

This script trains the RNN on text data and demonstrates:
- Training loop with loss tracking
- Text generation during training
- Gradient magnitude tracking (to see vanishing/exploding gradients)

Usage:
    python train.py

You can modify the hyperparameters below to experiment.
"""

import numpy as np
from rnn import RNN, create_dataset, get_batch

# =============================================================================
# HYPERPARAMETERS - Tune these!
# =============================================================================

HIDDEN_SIZE = 128      # Number of hidden units (memory capacity)
SEQ_LENGTH = 30        # Number of timesteps to unroll for BPTT
LEARNING_RATE = 0.05   # Learning rate for Adagrad
MAX_ITERS = 8000       # Total training iterations
PRINT_EVERY = 2000     # Print progress every N iterations
SAMPLE_EVERY = 4000    # Generate sample text every N iterations

# =============================================================================
# SAMPLE DATA - Shakespeare excerpt
# =============================================================================

# You can replace this with any text file
DATA = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.
"""

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train():
    """
    Main training loop.

    Returns:
        rnn: trained model
        history: training history (losses, gradient norms)
    """
    print("micro-rnn Training")
    print("=" * 60)

    # Create dataset
    char_to_idx, idx_to_char, data = create_dataset(DATA)
    vocab_size = len(char_to_idx)

    print(f"\nData length: {len(data)} characters")
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Sequence length: {SEQ_LENGTH}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("=" * 60)

    # Initialize model
    rnn = RNN(vocab_size, HIDDEN_SIZE, vocab_size)

    # Training state
    n = 0  # iteration counter
    p = 0  # data pointer

    # History for plotting
    history = {
        'loss': [],
        'smooth_loss': [],
        'grad_norms': {'W_xh': [], 'W_hh': [], 'W_hy': []},
        'iterations': []
    }

    # Initial loss (for smoothing)
    smooth_loss = -np.log(1.0 / vocab_size) * SEQ_LENGTH

    # Initial hidden state
    h_prev = np.zeros((HIDDEN_SIZE, 1))

    print("\nStarting training...\n")

    while n < MAX_ITERS:
        # Reset if we've gone through all data
        if p + SEQ_LENGTH + 1 >= len(data):
            p = 0
            h_prev = np.zeros((HIDDEN_SIZE, 1))

        # Get batch
        inputs, targets = get_batch(data, p, SEQ_LENGTH)

        # Forward pass
        loss, cache, h_prev = rnn.forward(inputs, targets, h_prev)

        # Backward pass
        grads = rnn.backward(inputs, targets, cache)

        # Clip gradients (prevent explosion)
        grads = rnn.clip_gradients(grads, max_norm=5)

        # Update parameters
        rnn.update_parameters(grads, LEARNING_RATE)

        # Track smooth loss (exponential moving average)
        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        # Record history periodically
        if n % 100 == 0:
            history['loss'].append(loss)
            history['smooth_loss'].append(smooth_loss)
            history['iterations'].append(n)

            # Track gradient norms (to visualize vanishing/exploding)
            for key in ['W_xh', 'W_hh', 'W_hy']:
                norm = np.linalg.norm(grads[key])
                history['grad_norms'][key].append(norm)

        # Print progress
        if n % PRINT_EVERY == 0:
            print(f"Iteration {n:5d} | Loss: {smooth_loss:.4f}")

            # Show gradient norms (useful for debugging)
            w_hh_norm = np.linalg.norm(grads['W_hh'])
            print(f"              | W_hh gradient norm: {w_hh_norm:.4f}")

        # Generate sample
        if n % SAMPLE_EVERY == 0 and n > 0:
            print("\n--- Sample ---")
            sample_indices = rnn.sample(inputs[0], h_prev, 200)
            sample_text = ''.join(idx_to_char[i] for i in sample_indices)
            print(sample_text)
            print("--- End Sample ---\n")

        # Move pointers
        p += SEQ_LENGTH
        n += 1

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Final loss: {smooth_loss:.4f}")
    print("=" * 60)

    # Final sample
    print("\n--- Final Sample (500 chars) ---")
    h = np.zeros((HIDDEN_SIZE, 1))
    seed = char_to_idx[DATA[0]]
    sample_indices = rnn.sample(seed, h, 500)
    sample_text = ''.join(idx_to_char[i] for i in sample_indices)
    print(sample_text)
    print("--- End Sample ---")

    return rnn, history, char_to_idx, idx_to_char


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    rnn, history, char_to_idx, idx_to_char = train()

    # Save history for visualization
    np.savez('training_history.npz',
             loss=history['loss'],
             smooth_loss=history['smooth_loss'],
             iterations=history['iterations'],
             grad_W_xh=history['grad_norms']['W_xh'],
             grad_W_hh=history['grad_norms']['W_hh'],
             grad_W_hy=history['grad_norms']['W_hy'])

    print("\n[DONE] Training history saved to training_history.npz")
    print("Run visualize.py to see training curves and gradient analysis.")
