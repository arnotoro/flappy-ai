from graphviz import Digraph

# Create a new directed graph
dqn_diagram = Digraph('DQN Architecture', filename='dqn_architecture_diagram', format='png')

# Set the graph layout to be top-to-bottom (TB) for better compactness
dqn_diagram.attr(rankdir='LR')

dqn_diagram.attr('node', shape='rect')

# Add nodes for the neural network
dqn_diagram.node('Input', 'State\n(3 features)', shape='ellipse', style='filled', fillcolor='lightblue')
dqn_diagram.node('Hidden1', 'Hidden Layer 1\n(128 neurons)', shape='ellipse', style='filled', fillcolor='lightgreen')
dqn_diagram.node('Hidden2', 'Hidden Layer 2\n(64 neurons)', shape='ellipse', style='filled', fillcolor='lightgreen')
dqn_diagram.node('Output', 'Q-values\n(2 actions)', shape='ellipse', style='filled', fillcolor='lightblue')

# Connect neural network layers
dqn_diagram.edge('Input', 'Hidden1')
dqn_diagram.edge('Hidden1', 'Hidden2')
dqn_diagram.edge('Hidden2', 'Output')

# Add DQN components
dqn_diagram.node('Env', 'Environment', shape='ellipse', style='filled', fillcolor='orange')
dqn_diagram.node('Policy', 'Epsilon-Greedy\nPolicy', shape='box', style='filled', fillcolor='lightyellow')
dqn_diagram.node('Buffer', 'Experience Buffer\n(Replay Memory)', shape='cylinder', style='filled', fillcolor='pink')
dqn_diagram.node('Trainer', 'Trainer', shape='box', style='filled', fillcolor='lightyellow')
dqn_diagram.node('Bellman', 'Target Q-value\n(Bellman Equation)', shape='box', style='filled', fillcolor='lightyellow')

# Connect workflow with individual edge definitions
dqn_diagram.edge('Env', 'Input', label='State')
dqn_diagram.edge('Output', 'Policy', label='Q-values')
dqn_diagram.edge('Policy', 'Env', label='Action')
dqn_diagram.edge('Policy', 'Buffer', label='Store (s, a, r, s\')')
dqn_diagram.edge('Buffer', 'Trainer', label='Sample Batch')
dqn_diagram.edge('Trainer', 'Bellman', label='Compute Target Q')
dqn_diagram.edge('Bellman', 'Trainer', label='Compute Loss')
dqn_diagram.edge('Trainer', 'Output', label='Update Parameters')

# Optional: Adjust node width and height for more compactness
dqn_diagram.attr('node', width='1.8', height='1.0')

# Render the graph
dqn_diagram.render('dqn_architecture_diagram', directory=r'C:\Users\arnot\Documents\flappy-ai', cleanup=True)
