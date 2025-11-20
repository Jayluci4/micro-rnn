# micro-rnn

A minimal implementation of Recurrent Neural Networks (RNNs) from scratch using only NumPy.

Part of the "Build AI From Scratch" series - bridging the gap between [micrograd](https://github.com/karpathy/micrograd) and modern architectures like Transformers.

## Why RNNs?

Before attention and transformers, RNNs were the go-to architecture for sequential data. Understanding RNNs is crucial because:

1. **Historical context** - Shows why attention mechanisms were invented
2. **Foundation for LSTMs/GRUs** - These are direct improvements on vanilla RNNs
3. **The vanishing gradient problem** - A fundamental challenge in deep learning

## What You'll Learn

- How RNNs process sequences using hidden states
- Forward pass through time
- Backpropagation Through Time (BPTT)
- Why gradients vanish/explode in deep networks
- Proper weight initialization for stable training
- Gradient clipping as a simple fix

## The Core Idea

```
h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
y_t = W_hy @ h_t + b_y
```

The magic is in `W_hh` - this **recurrent weight** connects the hidden state to itself across time, giving the network memory.

## Files

```
micro-rnn/
  rnn.py        # Core RNN implementation (~300 lines, heavily commented)
  train.py      # Training loop with character-level language modeling
  visualize.py  # Visualizations for gradients, hidden states, architecture
  demo.ipynb    # Interactive Jupyter notebook tutorial
  README.md     # This file
```

## Quick Start

### 1. Run the core tests

```bash
python rnn.py
```

This runs gradient checking to verify backprop is correct.

### 2. Train on text

```bash
python train.py
```

Trains the RNN on Shakespeare text and generates samples.

### 3. Visualize

```bash
python visualize.py
```

Creates visualizations showing:
- Training curves
- Vanishing gradient problem
- Hidden state dynamics
- RNN architecture diagram

### 4. Interactive notebook

```bash
jupyter notebook demo.ipynb
```

Step-by-step walkthrough of the concepts.

## The Vanishing Gradient Problem

This is the key insight this repo demonstrates.

When we backpropagate through time, gradients get multiplied by `W_hh` at each timestep:

```
gradient at t=0 ~ W_hh^T * W_hh^(T-1) * ... * W_hh^1
```

If eigenvalues of W_hh are:
- **< 1**: gradients shrink exponentially (vanishing)
- **> 1**: gradients grow exponentially (exploding)

This means:
- Early timesteps get tiny gradients
- RNN struggles to learn long-range dependencies
- "The quick brown fox jumps over the lazy ___" - hard to learn that "dog" relates to "fox"

**Solution**: LSTMs and GRUs use gates to control gradient flow, allowing gradients to pass through unchanged.

## Code Highlights

### Weight Initialization

```python
def init_weights(rows, cols):
    """
    Initialize weights with proper scaling for stable gradients.
    Scale by sqrt(1/fan_in) to keep variance stable across layers.
    """
    scale = np.sqrt(1.0 / cols)
    return np.random.randn(rows, cols) * scale
```

This keeps signal variance stable - without it, activations vanish or explode before training even starts.

### Forward Step

```python
def forward_step(self, x, h_prev):
    # Combine current input + previous memory
    h = np.tanh(
        np.dot(self.W_xh, x) +      # input contribution
        np.dot(self.W_hh, h_prev) +  # memory contribution
        self.b_h
    )

    # Output
    y = np.dot(self.W_hy, h) + self.b_y
    p = softmax(y)

    return h, y, p
```

### Backward Pass (BPTT)

```python
# Key part: gradient flows through W_hh
for t in reversed(range(seq_length)):
    # ... compute dh_raw ...

    # This multiplication causes vanishing/exploding gradients!
    dh_next = np.dot(self.W_hh.T, dh_raw)
```

### Gradient Clipping

```python
def clip_gradients(self, grads, max_norm=3):
    # Simple but effective - just cap the values
    for key, grad in grads.items():
        clipped[key] = np.clip(grad, -max_norm, max_norm)
    return clipped
```

## Hyperparameters

Default settings that work well:

```python
HIDDEN_SIZE = 128      # Memory capacity
SEQ_LENGTH = 30        # Timesteps for BPTT
LEARNING_RATE = 0.05   # For Adagrad optimizer
MAX_ITERS = 20000      # Training iterations
```

## Dependencies

Just NumPy and Matplotlib:

```bash
pip install numpy matplotlib
```

For the notebook:

```bash
pip install jupyter
```

## What's Next?

This is the first in a series bridging basic neural nets to modern AI:

1. **micro-rnn** (this repo) - Vanilla RNN, vanishing gradients
2. **micro-lstm** - Gates to control gradient flow
3. **micro-gru** - Simplified LSTM
4. **micro-embedding** - Word vectors from scratch
5. **micro-seq2seq** - Encoder-decoder architecture

Then you're ready for attention and transformers!

## References

- [Karpathy's min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086) - Original inspiration
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Colah's blog
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) - Karpathy's blog

## Philosophy

- Minimal code (~300 lines for core RNN)
- Heavy comments explaining every line
- Visualizations to build intuition
- First principles - build everything yourself
- No frameworks - just NumPy

## License

MIT
