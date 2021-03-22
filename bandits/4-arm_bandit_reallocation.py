import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.ion()
plt.style.use('fivethirtyeight')

############################################
# Simulation variables
############################################
num_days_experiment = 200
a1_mean = 4.025
a2_mean = 3.654
a3_mean = 4.312
a4_mean = 3.8475

var = 1

plot_on = True
num_sim = 10000
daily_budget = 100

beta = 0.01 # Pursuit Step Size

action_preferences = {'a1': 0.25, 'a2': 0.25, 'a3':0.25 , 'a4':0.25}
budget_pc_allocations_history = {'a1': [], 'a2': [], 'a3': [], 'a4': []}

# Generate the ROAS for the number of days the experiment is conducted for both the arms
a1_roas = [abs(i) for i in np.random.normal(a1_mean, var, num_days_experiment)]
a2_roas = [abs(i) for i in np.random.normal(a2_mean, var, num_days_experiment)]
a3_roas = [abs(i) for i in np.random.normal(a3_mean, var, num_days_experiment)]
a4_roas = [abs(i) for i in np.random.normal(a4_mean, var, num_days_experiment)]

roas = {'a1': a1_roas, 'a2': a2_roas, 'a3': a3_roas, 'a4': a4_roas}

# Plot the true roas distributions
if plot_on:
    sns.kdeplot(a1_roas)
    sns.kdeplot(a2_roas)
    sns.kdeplot(a3_roas)
    sns.kdeplot(a4_roas)
    plt.savefig("4-arm-kdeplot.png",dpi=200)
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
    plt.savefig("4-arm_bandit_reallocation.png",dpi=200)

