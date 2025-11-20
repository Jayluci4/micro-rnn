"""
micro-rnn: A minimal RNN implementation from scratch
=====================================================

This implements a vanilla RNN for character-level language modeling.
Built from first principles with only numpy - no frameworks.

Key concepts covered:
- RNN forward pass (how hidden states flow through time)
- Backpropagation Through Time (BPTT)
- Vanishing/exploding gradient problem
- Gradient clipping

Architecture:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_hy @ h_t + b_y

Where:
    x_t = input at time t (one-hot encoded character)
    h_t = hidden state at time t
    y_t = output logits at time t
"""

import numpy as np

# =============================================================================
# WEIGHT INITIALIZATION
# =============================================================================

def init_weights(rows, cols):
    """
    Initialize weights with proper scaling for stable gradients.

    Scale by sqrt(1/fan_in) to keep variance stable across layers.
    Without this, signals either vanish or explode as they pass through.

    Args:
        rows: output dimension
        cols: input dimension (fan_in)

    Returns:
        weight matrix with proper scaling
    """
    scale = np.sqrt(1.0 / cols)
    return np.random.randn(rows, cols) * scale


# =============================================================================
# RNN CLASS - The Core Implementation
# =============================================================================

class RNN:
    """
    Vanilla RNN for character-level language modeling.

    This is the simplest form of RNN - understanding this deeply
    will help you appreciate why LSTMs and GRUs were invented.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize RNN with random weights.

        Args:
            input_size: vocabulary size (number of unique characters)
            hidden_size: number of hidden units (the "memory" capacity)
            output_size: vocabulary size (predicting next character)

        Weight initialization uses proper scaling for stable gradients.
        See init_weights() function above.
        """
        self.hidden_size = hidden_size

        # =====================================================================
        # WEIGHTS - Properly scaled for stable training
        # =====================================================================

        # W_xh: input to hidden (how input affects hidden state)
        self.W_xh = init_weights(hidden_size, input_size)

        # W_hh: hidden to hidden (how previous hidden state affects current)
        # This is the "recurrent" part - it's what gives RNN memory
        self.W_hh = init_weights(hidden_size, hidden_size)

        # W_hy: hidden to output (how hidden state produces output)
        self.W_hy = init_weights(output_size, hidden_size)

        # =====================================================================
        # BIASES - Learnable offsets
        # =====================================================================

        # b_h: hidden bias
        self.b_h = np.zeros((hidden_size, 1))

        # b_y: output bias
        self.b_y = np.zeros((output_size, 1))

        # =====================================================================
        # MEMORY FOR ADAGRAD - Adaptive learning rates per parameter
        # =====================================================================
        # We accumulate squared gradients here for Adagrad optimizer
        self.mW_xh = np.zeros_like(self.W_xh)
        self.mW_hh = np.zeros_like(self.W_hh)
        self.mW_hy = np.zeros_like(self.W_hy)
        self.mb_h = np.zeros_like(self.b_h)
        self.mb_y = np.zeros_like(self.b_y)

    def forward_step(self, x, h_prev):
        """
        Single forward step of RNN.

        This is the core computation that happens at each timestep:
            h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            y_t = W_hy @ h_t + b_y

        Args:
            x: one-hot input vector, shape (input_size, 1)
            h_prev: previous hidden state, shape (hidden_size, 1)

        Returns:
            h: new hidden state
            y: output logits (unnormalized scores)
            p: output probabilities (softmax of y)

        Why tanh?
        - Outputs between -1 and 1
        - Zero-centered (unlike sigmoid)
        - But still suffers from vanishing gradients for extreme values
        """
        # Compute new hidden state
        # This combines: current input + previous memory + bias
        h = np.tanh(
            np.dot(self.W_xh, x) +      # input contribution
            np.dot(self.W_hh, h_prev) +  # memory contribution
            self.b_h                     # bias
        )

        # Compute output logits
        y = np.dot(self.W_hy, h) + self.b_y

        # Convert to probabilities via softmax
        # Subtract max for numerical stability (prevents overflow)
        p = np.exp(y - np.max(y))
        p = p / np.sum(p)

        return h, y, p

    def forward(self, inputs, targets, h_prev):
        """
        Forward pass through entire sequence + compute loss.

        Args:
            inputs: list of input indices (character indices)
            targets: list of target indices (next characters)
            h_prev: initial hidden state

        Returns:
            loss: cross-entropy loss over sequence
            cache: saved values needed for backward pass

        We save intermediate values in cache for backprop.
        This is the "forward pass" part of forward-backward algorithm.
        """
        # Storage for intermediate values (needed for backprop)
        xs = {}      # inputs (one-hot)
        hs = {}      # hidden states
        ys = {}      # output logits
        ps = {}      # probabilities

        # Initialize with previous hidden state
        hs[-1] = np.copy(h_prev)

        loss = 0

        # Process each timestep
        for t in range(len(inputs)):
            # One-hot encode input
            xs[t] = np.zeros((self.W_xh.shape[1], 1))
            xs[t][inputs[t]] = 1

            # Forward step
            hs[t], ys[t], ps[t] = self.forward_step(xs[t], hs[t-1])

            # Cross-entropy loss: -log(probability of correct character)
            # This is what we're trying to minimize
            loss += -np.log(ps[t][targets[t], 0])

        # Cache everything for backward pass
        cache = (xs, hs, ps)

        return loss, cache, hs[len(inputs)-1]

    def backward(self, inputs, targets, cache):
        """
        Backward pass: Backpropagation Through Time (BPTT).

        This is where the magic (and pain) happens.
        We compute gradients by going backward through time.

        Args:
            inputs: list of input indices
            targets: list of target indices
            cache: saved values from forward pass

        Returns:
            grads: dictionary of gradients for all parameters

        Key insight: gradients flow backward through the W_hh connections.
        This is why RNNs have vanishing/exploding gradient problems -
        gradients get multiplied by W_hh at each timestep.
        """
        xs, hs, ps = cache

        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        # This will accumulate gradient flowing back through hidden states
        dh_next = np.zeros_like(hs[0])

        # Go backward through time
        for t in reversed(range(len(inputs))):
            # =================================================================
            # OUTPUT LAYER GRADIENTS
            # =================================================================

            # Gradient of loss w.r.t. output (softmax + cross-entropy)
            # This elegant formula comes from calculus:
            # d(loss)/d(y) = p - one_hot(target)
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1  # This is the "error" at output

            # Gradient for W_hy: outer product of dy and h
            dW_hy += np.dot(dy, hs[t].T)
            db_y += dy

            # =================================================================
            # HIDDEN LAYER GRADIENTS
            # =================================================================

            # Gradient flows back from output AND from next timestep
            dh = np.dot(self.W_hy.T, dy) + dh_next

            # Backprop through tanh: d(tanh)/dx = 1 - tanh^2(x)
            # Since hs[t] = tanh(...), we have:
            dh_raw = (1 - hs[t] ** 2) * dh

            # Gradient for biases
            db_h += dh_raw

            # Gradient for W_xh
            dW_xh += np.dot(dh_raw, xs[t].T)

            # Gradient for W_hh
            dW_hh += np.dot(dh_raw, hs[t-1].T)

            # =================================================================
            # PASS GRADIENT TO PREVIOUS TIMESTEP
            # =================================================================
            # This is the critical part - gradient flows through W_hh
            # At each step, gradient gets multiplied by W_hh
            # If W_hh has eigenvalues > 1: exploding gradients
            # If W_hh has eigenvalues < 1: vanishing gradients
            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Pack gradients
        grads = {
            'W_xh': dW_xh, 'W_hh': dW_hh, 'W_hy': dW_hy,
            'b_h': db_h, 'b_y': db_y
        }

        return grads

    def clip_gradients(self, grads, max_norm=3):
        """
        Clip gradients to prevent exploding gradients.

        Without this, gradients can become huge and training diverges.
        This is a simple but effective solution - just cap the values.

        Args:
            grads: dictionary of gradients
            max_norm: maximum allowed value (clips to [-max_norm, max_norm])

        Returns:
            clipped_grads: gradients with values clipped

        Note: This doesn't solve vanishing gradients - that requires
        architectural changes (LSTM, GRU) which we'll cover next.
        """
        clipped = {}
        for key, grad in grads.items():
            clipped[key] = np.clip(grad, -max_norm, max_norm)
        return clipped

    def update_parameters(self, grads, learning_rate=0.1):
        """
        Update parameters using Adagrad optimizer.

        Adagrad adapts learning rate per-parameter based on history:
        - Frequently updated params get smaller learning rates
        - Rarely updated params get larger learning rates

        Update rule:
            memory += grad^2
            param -= learning_rate * grad / sqrt(memory + eps)

        Args:
            grads: dictionary of gradients
            learning_rate: base learning rate
        """
        eps = 1e-8  # small constant to prevent division by zero

        # Update each parameter with Adagrad
        for param, dparam, mem in [
            (self.W_xh, grads['W_xh'], 'mW_xh'),
            (self.W_hh, grads['W_hh'], 'mW_hh'),
            (self.W_hy, grads['W_hy'], 'mW_hy'),
            (self.b_h, grads['b_h'], 'mb_h'),
            (self.b_y, grads['b_y'], 'mb_y'),
        ]:
            # Get memory array
            m = getattr(self, mem)

            # Accumulate squared gradient
            m += dparam ** 2

            # Update parameter (in-place)
            param -= learning_rate * dparam / np.sqrt(m + eps)

    def sample(self, seed_idx, h, length):
        """
        Generate text by sampling from the model.

        Args:
            seed_idx: starting character index
            h: initial hidden state
            length: number of characters to generate

        Returns:
            indices: list of generated character indices

        We sample from the probability distribution at each step,
        rather than always picking the most likely character.
        This adds diversity to the generated text.
        """
        # One-hot encode seed
        x = np.zeros((self.W_xh.shape[1], 1))
        x[seed_idx] = 1

        indices = []

        for _ in range(length):
            # Forward step
            h, y, p = self.forward_step(x, h)

            # Sample from distribution
            # np.random.choice picks index based on probabilities
            idx = np.random.choice(range(p.shape[0]), p=p.ravel())
            indices.append(idx)

            # Prepare input for next step
            x = np.zeros_like(x)
            x[idx] = 1

        return indices


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_dataset(text):
    """
    Create character-level dataset from text.

    Args:
        text: input text string

    Returns:
        char_to_idx: dict mapping characters to indices
        idx_to_char: dict mapping indices to characters
        data: list of character indices

    We convert text to indices for efficient processing.
    The vocabulary is all unique characters in the text.
    """
    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {''.join(chars)}")

    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert text to indices
    data = [char_to_idx[ch] for ch in text]

    return char_to_idx, idx_to_char, data


def get_batch(data, batch_start, seq_length):
    """
    Get a batch of training data.

    Args:
        data: full dataset (list of indices)
        batch_start: starting position in data
        seq_length: length of sequence to extract

    Returns:
        inputs: input sequence (characters 0 to seq_length-1)
        targets: target sequence (characters 1 to seq_length)

    For language modeling, target is the next character.
    So if input is "hell", target is "ello".
    """
    inputs = data[batch_start:batch_start + seq_length]
    targets = data[batch_start + 1:batch_start + seq_length + 1]
    return inputs, targets


# =============================================================================
# GRADIENT CHECKING - Verify backprop is correct
# =============================================================================

def gradient_check(rnn, inputs, targets, h_prev):
    """
    Numerical gradient checking to verify backprop implementation.

    Computes gradients numerically using finite differences:
        df/dx = (f(x+h) - f(x-h)) / (2h)

    Then compares with analytical gradients from backprop.
    If they match, our backprop is correct!

    This is slow but essential for debugging.
    """
    print("Running gradient check...")

    # Get analytical gradients
    loss, cache, _ = rnn.forward(inputs, targets, h_prev)
    grads = rnn.backward(inputs, targets, cache)

    # Check each parameter
    for param_name in ['W_xh', 'W_hh', 'W_hy', 'b_h', 'b_y']:
        param = getattr(rnn, param_name)
        grad = grads[param_name]

        # Check a few random elements
        for _ in range(5):
            idx = tuple(np.random.randint(0, s) for s in param.shape)

            # Compute numerical gradient
            h = 1e-5
            old_val = param[idx]

            param[idx] = old_val + h
            loss_plus, _, _ = rnn.forward(inputs, targets, h_prev)

            param[idx] = old_val - h
            loss_minus, _, _ = rnn.forward(inputs, targets, h_prev)

            param[idx] = old_val  # restore

            numerical_grad = (loss_plus - loss_minus) / (2 * h)
            analytical_grad = grad[idx]

            # Relative error
            rel_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + abs(analytical_grad) + 1e-8)

            if rel_error > 1e-5:
                print(f"[ERROR] {param_name}{idx}: numerical={numerical_grad:.6f}, analytical={analytical_grad:.6f}, error={rel_error:.6f}")
            else:
                print(f"[OK] {param_name}{idx}: error={rel_error:.2e}")

    print("Gradient check complete!")


if __name__ == "__main__":
    # Quick test
    print("micro-rnn: Minimal RNN from scratch")
    print("=" * 50)

    # Simple test data
    text = "hello world. hello rnn. hello deep learning."
    char_to_idx, idx_to_char, data = create_dataset(text)

    # Create RNN
    vocab_size = len(char_to_idx)
    hidden_size = 32
    rnn = RNN(vocab_size, hidden_size, vocab_size)

    # Test forward pass
    inputs, targets = get_batch(data, 0, 10)
    h_prev = np.zeros((hidden_size, 1))

    loss, cache, h_new = rnn.forward(inputs, targets, h_prev)
    print(f"\nTest forward pass - Loss: {loss:.4f}")

    # Test backward pass
    grads = rnn.backward(inputs, targets, cache)
    print(f"Backward pass complete - computed {len(grads)} gradients")

    # Gradient check
    print()
    gradient_check(rnn, inputs, targets, h_prev)

    print("\n[DONE] Basic tests passed!")
