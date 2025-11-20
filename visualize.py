"""
Visualizations for micro-rnn
=============================

This script creates visualizations to understand:
1. Training loss curves
2. Gradient flow through time (vanishing gradient problem!)
3. Hidden state dynamics

Run this after train.py to see the plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from rnn import RNN, create_dataset, get_batch

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================

def plot_training_history(filename='training_history.npz'):
    """
    Plot training loss and gradient norms over time.

    This helps you understand:
    - Is the model learning? (loss going down)
    - Are gradients healthy? (not too big, not too small)
    """
    try:
        data = np.load(filename)
    except FileNotFoundError:
        print(f"[ERROR] {filename} not found. Run train.py first!")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Loss over time
    ax1 = axes[0]
    ax1.plot(data['iterations'], data['loss'], alpha=0.3, label='Loss')
    ax1.plot(data['iterations'], data['smooth_loss'], label='Smooth Loss', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gradient norms
    ax2 = axes[1]
    ax2.plot(data['iterations'], data['grad_W_xh'], label='W_xh', alpha=0.7)
    ax2.plot(data['iterations'], data['grad_W_hh'], label='W_hh (recurrent)', linewidth=2)
    ax2.plot(data['iterations'], data['grad_W_hy'], label='W_hy', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Magnitudes (W_hh is the recurrent weight)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("[OK] Saved training_curves.png")
    plt.show()


# =============================================================================
# VISUALIZE VANISHING GRADIENT PROBLEM
# =============================================================================

def visualize_vanishing_gradients():
    """
    Demonstrate the vanishing gradient problem.

    Key insight: When we backpropagate through time, gradients get
    multiplied by W_hh at each step. This causes:
    - Vanishing: if eigenvalues < 1, gradient shrinks exponentially
    - Exploding: if eigenvalues > 1, gradient grows exponentially

    This visualization shows gradient magnitude at each timestep.
    """
    print("\nVisualizing Vanishing Gradient Problem")
    print("=" * 50)

    # Create a simple RNN
    vocab_size = 10
    hidden_size = 50
    rnn = RNN(vocab_size, hidden_size, vocab_size)

    # Create sequences of different lengths
    sequence_lengths = [5, 10, 25, 50, 100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Storage for gradient flow analysis
    all_grad_flows = []

    for seq_len in sequence_lengths:
        # Random input/target sequence
        inputs = list(np.random.randint(0, vocab_size, seq_len))
        targets = list(np.random.randint(0, vocab_size, seq_len))

        h_prev = np.zeros((hidden_size, 1))

        # Forward pass
        loss, cache, _ = rnn.forward(inputs, targets, h_prev)
        xs, hs, ps = cache

        # Manual backward pass to track gradient at each timestep
        dh_next = np.zeros((hidden_size, 1))
        grad_magnitudes = []

        for t in reversed(range(seq_len)):
            # Output gradient
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1

            # Hidden gradient
            dh = np.dot(rnn.W_hy.T, dy) + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh

            # Track magnitude
            grad_magnitudes.append(np.linalg.norm(dh_raw))

            # Pass to previous timestep
            dh_next = np.dot(rnn.W_hh.T, dh_raw)

        # Reverse to get chronological order
        grad_magnitudes = grad_magnitudes[::-1]
        all_grad_flows.append((seq_len, grad_magnitudes))

    # Plot 1: Gradient magnitude over timesteps
    ax1 = axes[0]
    for seq_len, grad_mags in all_grad_flows:
        timesteps = range(seq_len)
        ax1.plot(timesteps, grad_mags, label=f'seq_len={seq_len}', marker='.')

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Gradient Magnitude')
    ax1.set_title('Gradient Flow Through Time\n(Notice how gradients vanish for early timesteps!)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Ratio of first/last gradient (measures vanishing)
    ax2 = axes[1]
    seq_lens = []
    ratios = []

    for seq_len, grad_mags in all_grad_flows:
        if grad_mags[0] > 0:
            ratio = grad_mags[-1] / grad_mags[0]  # last / first
            seq_lens.append(seq_len)
            ratios.append(ratio)

    ax2.bar(range(len(seq_lens)), ratios, tick_label=seq_lens)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Gradient Ratio (last/first)')
    ax2.set_title('Vanishing Gradient Effect\n(Higher ratio = more vanishing)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('vanishing_gradients.png', dpi=150)
    print("[OK] Saved vanishing_gradients.png")
    plt.show()

    print("\nKey Observation:")
    print("- Gradients for early timesteps are MUCH smaller than later ones")
    print("- This means early inputs have little effect on learning")
    print("- This is the VANISHING GRADIENT problem!")
    print("- Solution: LSTM/GRU use gates to control gradient flow")


# =============================================================================
# VISUALIZE HIDDEN STATE DYNAMICS
# =============================================================================

def visualize_hidden_states():
    """
    Visualize how hidden states evolve over a sequence.

    This shows:
    - How information flows through the RNN
    - The "memory" patterns in hidden units
    """
    print("\nVisualizing Hidden State Dynamics")
    print("=" * 50)

    # Sample text
    text = "The quick brown fox jumps over the lazy dog."
    char_to_idx, idx_to_char, data = create_dataset(text)
    vocab_size = len(char_to_idx)

    # Create and train a small RNN
    hidden_size = 20
    rnn = RNN(vocab_size, hidden_size, vocab_size)

    # Quick training (just to get some patterns)
    h = np.zeros((hidden_size, 1))
    for _ in range(100):
        if len(data) > 1:
            inputs = data[:-1]
            targets = data[1:]
            loss, cache, h = rnn.forward(inputs, targets, h)
            grads = rnn.backward(inputs, targets, cache)
            grads = rnn.clip_gradients(grads)
            rnn.update_parameters(grads, 0.1)

    # Now visualize hidden states
    h = np.zeros((hidden_size, 1))
    hidden_states = []

    for idx in data:
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        h, _, _ = rnn.forward_step(x, h)
        hidden_states.append(h.flatten())

    hidden_states = np.array(hidden_states)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(hidden_states.T, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Activation')

    # Label x-axis with characters
    ax.set_xticks(range(len(text)))
    ax.set_xticklabels(list(text), fontsize=8)
    ax.set_xlabel('Character in sequence')
    ax.set_ylabel('Hidden Unit')
    ax.set_title('Hidden State Evolution\n(Each column is hidden state after reading that character)')

    plt.tight_layout()
    plt.savefig('hidden_states.png', dpi=150)
    print("[OK] Saved hidden_states.png")
    plt.show()

    print("\nObservations:")
    print("- Different hidden units activate for different characters")
    print("- Some units track specific patterns (like spaces, vowels)")
    print("- The hidden state is the RNN's 'memory' of the sequence")


# =============================================================================
# VISUALIZE RNN ARCHITECTURE
# =============================================================================

def visualize_architecture():
    """
    Create a simple diagram of RNN architecture.
    """
    print("\nVisualizing RNN Architecture")
    print("=" * 50)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Draw unrolled RNN for 4 timesteps
    timesteps = 4
    x_positions = np.linspace(1, 10, timesteps)

    for i, x in enumerate(x_positions):
        # Input
        circle_in = plt.Circle((x, 1), 0.3, color='lightblue', ec='black')
        ax.add_patch(circle_in)
        ax.text(x, 1, f'x{i}', ha='center', va='center', fontsize=10)

        # Hidden
        circle_h = plt.Circle((x, 3), 0.4, color='lightgreen', ec='black')
        ax.add_patch(circle_h)
        ax.text(x, 3, f'h{i}', ha='center', va='center', fontsize=10)

        # Output
        circle_out = plt.Circle((x, 5), 0.3, color='lightyellow', ec='black')
        ax.add_patch(circle_out)
        ax.text(x, 5, f'y{i}', ha='center', va='center', fontsize=10)

        # Arrows: input to hidden
        ax.annotate('', xy=(x, 2.6), xytext=(x, 1.3),
                   arrowprops=dict(arrowstyle='->', color='blue'))

        # Arrows: hidden to output
        ax.annotate('', xy=(x, 4.7), xytext=(x, 3.4),
                   arrowprops=dict(arrowstyle='->', color='orange'))

        # Arrows: hidden to hidden (recurrent)
        if i < timesteps - 1:
            next_x = x_positions[i + 1]
            ax.annotate('', xy=(next_x - 0.4, 3), xytext=(x + 0.4, 3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Labels
    ax.text(0.3, 1, 'Input', ha='center', va='center', fontsize=10)
    ax.text(0.3, 3, 'Hidden', ha='center', va='center', fontsize=10)
    ax.text(0.3, 5, 'Output', ha='center', va='center', fontsize=10)

    # Legend
    ax.plot([], [], 'b->', label='W_xh (input to hidden)')
    ax.plot([], [], 'r->', label='W_hh (hidden to hidden) - RECURRENT', linewidth=2)
    ax.plot([], [], color='orange', marker='>', linestyle='-', label='W_hy (hidden to output)')

    ax.set_xlim(-0.5, 11)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RNN Unrolled Through Time\n(Red arrows show recurrent connections)', fontsize=12)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('rnn_architecture.png', dpi=150)
    print("[OK] Saved rnn_architecture.png")
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("micro-rnn Visualizations")
    print("=" * 60)

    # All visualizations
    visualize_architecture()
    visualize_vanishing_gradients()
    visualize_hidden_states()

    # Plot training history if available
    try:
        plot_training_history()
    except Exception as e:
        print(f"\n[NOTE] Could not plot training history: {e}")
        print("Run train.py first to generate training data.")

    print("\n[DONE] All visualizations complete!")
    print("Check the generated PNG files for the plots.")
