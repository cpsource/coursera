# After training
trained_policy = PolicyNetwork(obs_dim, 128, n_actions)
trained_policy.load_state_dict(torch.load('trained_model.pth'))

# In production
def make_decision(current_state):
    with torch.no_grad():
        action_probs = trained_policy(current_state)
        return torch.argmax(action_probs).item()  # Take best action

# Real-world example: Trading bot
market_state = get_market_data()
action = make_decision(market_state)  # 0=hold, 1=buy, 2=sell
execute_trade(action)

