With parameters
```
batch_size = 32     
max_iters = 3000     # Total number of training iterations
eval_interval = 300  # How often to evaluate the model on validation data
learning_rate = 3e-3 # Initial learning rate for the Adam optimizer
block_size = 128     # Maximum sequence length the model can process (context window)
n_embd = 128        # Dimension of embedding vectors and hidden layers
n_head = 1          # Number of attention heads in multi-head attention
n_layer = 1         # Number of transformer blocks stacked together
dropout = 0.2
```
Outputs are 
```
Input: "23"
Output: "23:23 And it came the saw it call against of the land that the mandment of they which with to his of m"

Input: " "
Output: "did came; and baptized a that were away but the churse behole the Lord of the were againto press the"
Output: "take unto the say heard commany had and the this brid behold, that hand among the come their way man"
```

With parameters
```
block_size = 256     # Maximum sequence length the model can process (context window)
n_embd = 256        # Dimension of embedding vectors and hidden layers
n_head = 1          # Number of attention heads in multi-head attention
n_layer = 2         # Number of transformer blocks stacked together
dropout = 0.2   
```
Outputs are
```
Input: "23"
Output: "23:19 s to people; new the cousla, and my alst at the Lamanits the orit haven.

1::9 Yeacording to Beh"
Input: " "
Output: "the words thausht they saw they ares st trestly that a children th of thout their with becomenals. A"
Output: "on hat toracited th the prissnnites; fore to be word he theire shold, als tar be als the age prosent"
```

Interestingly, the larger model is not as good!