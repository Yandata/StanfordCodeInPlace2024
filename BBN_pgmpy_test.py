#!/usr/bin/env python
# coding: utf-8

# Version 7 
# - build a BBN of 9 nodes using pgmpy package, 
# - define the CPDs (Conditional Probability Distributions), 
# - visualize the Bayesian Network using networkx and pyvis (interactive graph)
# - add labels using marginal probability values to the nodes
# - compute and display the marginal probabilities for each node
# - print the CPDs to verify
# - automate calculations and summarize results: to pass global parameter values, and to calculate marginal probabilities for a list of scenarios (using a dictionary)

# In[1]:


# Step 1: Install necessary packages
get_ipython().system('pip install pgmpy pyvis')

# Step 2: Import necessary classes from pgmpy and other libraries
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
from pyvis.network import Network


# In[10]:


# Define global parameters
P_event_A = 0.1
P_event_B = 0.01

# Create a Bayesian Network model
model = BayesianNetwork([
    ('X1', 'X1_event'), 
    ('X1_event', 'S1'), 
    ('X2', 'X2_event'), 
    ('X2_event', 'S2'), 
    ('S1', 'S2'), 
    ('X3', 'X3_event'), 
    ('X3_event', 'S3'), 
    ('S2', 'S3')
])

# CPD for X1
cpd_X1 = TabularCPD(
    variable='X1', 
    variable_card=2, 
    values=[[0.05], [0.95]], 
    state_names={'X1': ['A', 'B']}
)

# CPD for X1_event
cpd_X1_event = TabularCPD(
    variable='X1_event', 
    variable_card=2,
    values=[
        [P_event_A, P_event_B],  # P(X1_event=event | X1=A), P(X1_event=event | X1=B)
        [1 - P_event_A, 1 - P_event_B]  # P(X1_event=non-event | X1=A), P(X1_event=non-event | X1=B)
    ],
    evidence=['X1'], 
    evidence_card=[2], 
    state_names={'X1_event': ['event', 'non-event'], 'X1': ['A', 'B']}
)

# CPD for S1
cpd_S1 = TabularCPD(variable='S1', variable_card=2,
                          values=[[1, 0.1],  # P(S1=No | X1_event=event), P(S1=no | X1_event=non-event)
                                  [0, 0.9]],  # P(S1=Yes | X1_event=event), P(S1=Yes | X1_event=non-event)
                          evidence=['X1_event'], evidence_card=[2], state_names={'S1': ['No', 'Yes'], 'X1_event': ['event', 'non-event']})

# CPD for X2
cpd_X2 = TabularCPD(
    variable='X2', 
    variable_card=2, 
    values=[[0.1], [0.9]], 
    state_names={'X2': ['A', 'B']}
)

# CPD for X2_event
cpd_X2_event = TabularCPD(
    variable='X2_event', 
    variable_card=2,
    values=[
        [P_event_A, P_event_B],  # P(X2_event=event | X2=A), P(X2_event=event | X2=B)
        [1 - P_event_A, 1 - P_event_B]  # P(X2_event=non-event | X2=A), P(X2_event=non-event | X2=B)
    ],
    evidence=['X2'], 
    evidence_card=[2], 
    state_names={'X2_event': ['event', 'non-event'], 'X2': ['A', 'B']}
)

# CPD for S2
cpd_S2 = TabularCPD(
    variable='S2', 
    variable_card=2,
    values=[
        [1, 0.3, 0, 0],  # P(S2=No | X2_event=event, S1=No), etc.
        [0, 0.7, 1, 1]   # P(S2=Yes | X2_event=event, S1=No), etc.
    ],
    evidence=['S1', 'X2_event'], 
    evidence_card=[2, 2], 
    state_names={'S2': ['No', 'Yes'], 'S1': ['No', 'Yes'], 'X2_event': ['event', 'non-event']}
)

# CPD for X3
cpd_X3 = TabularCPD(
    variable='X3', 
    variable_card=2, 
    values=[[0.2], [0.8]], 
    state_names={'X3': ['A', 'B']}
)

# CPD for X3_event
cpd_X3_event = TabularCPD(
    variable='X3_event', 
    variable_card=2,
    values=[
        [P_event_A, P_event_B],  # P(X3_event=event | X3=A), etc.
        [1 - P_event_A, 1 - P_event_B]  # P(X3_event=non-event | X3=A), etc.
    ],
    evidence=['X3'], 
    evidence_card=[2], 
    state_names={'X3_event': ['event', 'non-event'], 'X3': ['A', 'B']}
)

# CPD for S3
cpd_S3 = TabularCPD(
    variable='S3', 
    variable_card=2,
    values=[
        [1, 0.5, 0, 0],  # P(S3=No | S2=No, X3_event=event), etc.
        [0, 0.5, 1, 1]   # P(S3=Yes | S2=No, X3_event=event), etc.
    ],
    evidence=['S2', 'X3_event'], 
    evidence_card=[2, 2], 
    state_names={'S3': ['No', 'Yes'], 'S2': ['No', 'Yes'], 'X3_event': ['event', 'non-event']}
)


# In[11]:


# Step 5: Add the CPDs to the model
model.add_cpds(cpd_X1, cpd_X1_event, cpd_S1, cpd_X2, cpd_X2_event, cpd_S2, cpd_X3, cpd_X3_event, cpd_S3)

# Check if the model is valid
assert model.check_model(), "Model is not valid"

# Step 6: Perform inference to get the marginal probabilities
inference = VariableElimination(model)


# In[12]:


# List of nodes for which to calculate marginal probabilities
nodes = ['X1', 'X1_event', 'S1', 'X2', 'X2_event', 'S2', 'X3', 'X3_event', 'S3']

# Dictionary to hold marginal probabilities for each node
marginals = {}

# Loop through each node to calculate marginal probabilities
for node in nodes:
    marginals[node] = inference.query(variables=[node], joint=False)[node]

# Step 7: Create a graph from the Bayesian Network model using networkx
G = nx.DiGraph()
G.add_edges_from(model.edges())

# Step 8: Add marginal probabilities as labels to the nodes
node_labels = {}
for node in nodes:
    label = f"{node}\n" + "\n".join([f"{state}: {prob:.6f}" for state, prob in zip(marginals[node].state_names[node], marginals[node].values)])
    node_labels[node] = label


# In[13]:


# Visualize the graph using pyvis
net = Network(notebook=True, height="700px", width="100%")
net.from_nx(G)

# Add labels to the nodes

# The following two lines of code set the title attribute, which requires hovering to show the label
# for node, label in node_labels.items():
#    net.get_node(node)['title'] = label

# Set the label Attribute:
# Instead of setting the title attribute, which requires hovering, the label attribute of each node is set directly to display the marginal probabilities permanently on the graph.
# Add labels to the nodes
for node, label in node_labels.items():
    net.get_node(node)['label'] = label  # Set the label attribute
    
net.show("bayesian_network.html")


# In[14]:


# Marginal probability for node S2
marginal_S2 = inference.query(variables=['S2'], joint=False)
print("Marginal Probability of S2:")
print(marginal_S2['S2'])

# Print the CPDs to verify
print("CPD of X1_event:")
print(cpd_X1_event)
print("\nCPD of X2_event:")
print(cpd_X2_event)
print("\nCPD of X3_event:")
print(cpd_X3_event)


# In[15]:


print(model.nodes())
print(model.edges())


# In[16]:


# The following code uses nested loops to iterate through all combinations of states for X1, X2, and X3. For each combination, it performs the query and stores the result in a dictionary. Finally, it prints the marginal probability of S3 given various combinations of of states for X1, X2, and X3.

# Define the states for X1, X2, X3
states = ['A', 'B']

# Initialize a dictionary to store the results
results = {}

# Loop through all combinations of states for X1, X2, X3
for x1 in states:
    for x2 in states:
        for x3 in states:
            evidence = {'X1': x1, 'X2': x2, 'X3': x3}
            query_result = inference.query(variables=['S3'], evidence=evidence)
            key = f"{x1}{x2}{x3}"
            results[key] = query_result.values[0]

# Print the results
for key, value in results.items():
    print(f"Probability of S3='No' given {key}: {value:.6f}")

