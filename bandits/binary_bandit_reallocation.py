import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
plt.style.use('fivethirtyeight')

############################################
# Simulation variables
############################################
num_days_experiment = 100
control_mean = 3.0
exp_mean = 3.8
control_var = 1.0
exp_var = 0.8
plot_on = True
num_sim = 10000
daily_budget = 100


beta = 0.01 # Pursuit Step Size
action_preferences = {'control': 0.5, 'exp': 0.5}
# budget_pc_allocations = {'control': 0.5, 'exp': 0.5}
budget_pc_allocations_history = {'control': [], 'exp': []}

# Generate the ROAS for the number of days the experiment is conducted for both the arms
control_roas = [abs(i) for i in np.random.normal(control_mean, control_var, num_days_experiment)]
experiment_roas = [abs(i) for i in np.random.normal(exp_mean, exp_var, num_days_experiment)]

roas = {'control': control_roas, 'exp': experiment_roas}

# Plot the true roas distributions
if plot_on:
    sns.kdeplot(control_roas)
    sns.kdeplot(experiment_roas)
    plt.savefig("binary_roas_kdeplot.png")
    plt.cla()
    plt.clf()

daily_budgets = [daily_budget] * num_days_experiment

for i, budget in enumerate(daily_budgets):
    # Step 1 : Sample best action 'num_sim' times using action preferences
    sample = list(np.random.choice(list(action_preferences.keys()), size=num_sim, replace=True,
                                   p=list(action_preferences.values())))

    # Step 2 : Get the budget allocation for the day
    for choice in list(action_preferences.keys()):
        # budget_pc_allocations[choice] = (sample.count(choice) / len(sample))
        budget_pc_allocations_history[choice].append((sample.count(choice) / len(sample)))

    # Step 3: Get the ROAS for the day
    best_roas = -100
    for choice in list(action_preferences.keys()):
        if roas[choice][i] > best_roas:
            best_roas = roas[choice][i]
            best_arm = choice

    # Step 4: Update the action preferences of each arm
    for choice in list(action_preferences.keys()):
        if choice == best_arm:
            action_preferences[choice] = max(0, action_preferences[choice] + beta * (
                        1 - budget_pc_allocations_history[choice][-1]))
        else:
            action_preferences[choice] = max(action_preferences[choice] + beta * (
                        0 - budget_pc_allocations_history[choice][-1]),0)

for choice in list(action_preferences.keys()):
    plt.plot(budget_pc_allocations_history[choice])
    plt.savefig("binary_bandit_reallocation.png",dpi=200)    

