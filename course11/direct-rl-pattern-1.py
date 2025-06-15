# NOT this (ranking other outputs):
lower_network_outputs = [output1, output2, output3]
rankings = rl_network.rank(lower_network_outputs)

# But THIS (direct decision making):
current_situation = get_current_state()
best_action = rl_network.decide(current_situation)
