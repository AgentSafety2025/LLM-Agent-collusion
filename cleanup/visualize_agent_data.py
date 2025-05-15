import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
from matplotlib.patches import Patch

# Load agent and experiment data
agent_df = pd.read_csv('agent_analysis_detailed.csv')
trial_df = pd.read_csv('experiment_analysis_detailed.csv')

# Ensure trial_df is sorted by unique_trial_id
trial_df = trial_df.sort_values('unique_trial_id')
agent_df = agent_df.sort_values('unique_trial_id')

# Debug: Print unique trial IDs in both DataFrames
print("All unique trial IDs from trial_df:", trial_df['unique_trial_id'].unique())
print("All unique trial IDs from agent_df:", agent_df['unique_trial_id'].unique())
all_trial_ids = list(trial_df['unique_trial_id'].unique())
print("All unique trial IDs (sorted):", all_trial_ids)
print("Total number of unique trials:", len(all_trial_ids))

# Model name mapping from raw to display names
MODEL_NAME_MAP = {
    'DeepSeek-R1': 'DeepSeek-R1',
    'Deepseek-r1': 'DeepSeek-R1',
    'claude-3-7-sonnet-20250219': 'Claude-3.7-Sonnet',
    'gemini-2.0-flash-001': 'Gemini-2.0-Flash',
    'gpt-4o-mini': 'o4-mini'
}

# Apply mapping to agent_df
agent_df['model'] = agent_df['model'].map(MODEL_NAME_MAP)

# Get all unique trial IDs, sorted
num_agents = 4
num_collusive_trials = 5
collusive_trial_ids = all_trial_ids[-num_collusive_trials:]
noncollusive_trial_ids = all_trial_ids[:-num_collusive_trials]

# Select rows for collusive and non-collusive trials
collusive_trial_rows = trial_df[trial_df['unique_trial_id'].isin(collusive_trial_ids)]
collusive_agent_rows = agent_df[agent_df['unique_trial_id'].isin(collusive_trial_ids)]
noncollusive_trial_rows = trial_df[trial_df['unique_trial_id'].isin(noncollusive_trial_ids)]
noncollusive_agent_rows = agent_df[agent_df['unique_trial_id'].isin(noncollusive_trial_ids)]

# Debug: Print non-collusive agent rows and metrics
print("Non-collusive agent rows:")
print(noncollusive_agent_rows)
print("Non-collusive trial IDs:", noncollusive_trial_ids)
print("Unique models in non-collusive agent rows:", noncollusive_agent_rows['model'].unique())

# Use these rows for all collusive metrics and plotting
# Redefine collusive_bar_labels, etc., as before
# Define model order and colors
ordered_models = ['DeepSeek-R1', 'Claude-3.7-Sonnet', 'Gemini-2.0-Flash', 'o4-mini']
colors_non_collusive = {
    'DeepSeek-R1': '#8F7289',
    'Claude-3.7-Sonnet': '#ED7B61',
    'Gemini-2.0-Flash': '#29A15C',
    'o4-mini': '#529AC9'
}
colors_collusive = {
    'DeepSeek-R1': '#D5B2BD',
    'Claude-3.7-Sonnet': '#F6B09C',
    'Gemini-2.0-Flash': '#9CCBA7',
    'o4-mini': '#99BADF'
}

# Helper: get metrics for each model, collusive/non-collusive
def get_model_metrics(agent_df, trial_numbers):
    metrics = {}
    for m in ordered_models:
        subset = agent_df[(agent_df['model'] == m) & (agent_df['unique_trial_id'].isin(trial_numbers))]
        metrics[m] = {
            'victories': subset['is_winner'].sum(),
            'signals_sent': subset['alliance_signals_sent'].sum(),
            'alliances_formed': subset['alliances_formed'].sum(),
            'river_cleaning': subset['river_cleaning_actions'].sum(),
            'zap_actions': subset['zap_actions'].sum(),
            'avg_score': subset['final_score'].mean() if not subset.empty else 0
        }
    return metrics

collusive_metrics = get_model_metrics(collusive_agent_rows, collusive_trial_ids)
noncollusive_metrics = get_model_metrics(noncollusive_agent_rows, noncollusive_trial_ids)

# Debug: Print non-collusive metrics
print("Non-collusive metrics:")
print(noncollusive_metrics)

# --- Grouped Bar Plots ---
bar_metrics = ['victories', 'river_cleaning', 'zap_actions', 'avg_score']
bar_titles = [
    'Victories', 'River Cleaning', 'Zap Actions', 'Average Score'
]
bar_ylabels = [
    'Count', 'Count', 'Count', 'Score'
]

fig, axes = plt.subplots(1, 4, figsize=(20, 6))
bar_width = 0.35
x = np.arange(len(ordered_models))

# For legend: organize by model, both conditions side by side
legend_elements = []
for m in ordered_models:
    legend_elements.append(Patch(facecolor=colors_non_collusive[m], label=f'{m} (Non-Collusive)', alpha=0.8))
    legend_elements.append(Patch(facecolor=colors_collusive[m], label=f'{m} (Collusive)', alpha=0.8))

for idx, metric in enumerate(bar_metrics):
    ax = axes[idx]
    collusive_vals = [collusive_metrics.get(m, {}).get(metric, 0) for m in ordered_models]
    noncollusive_vals = [noncollusive_metrics.get(m, {}).get(metric, 0) for m in ordered_models]
    collusive_vals = [v if not np.isnan(v) else 0 for v in collusive_vals]
    noncollusive_vals = [v if not np.isnan(v) else 0 for v in noncollusive_vals]
    # Bar colors by model
    nc_colors = [colors_non_collusive[m] for m in ordered_models]
    c_colors = [colors_collusive[m] for m in ordered_models]
    ax.bar(x - bar_width/2, noncollusive_vals, width=bar_width, label='Non-Collusive', color=nc_colors, alpha=0.8)
    ax.bar(x + bar_width/2, collusive_vals, width=bar_width, label='Collusive', color=c_colors, alpha=0.8)
    for i, v in enumerate(noncollusive_vals):
        ax.text(x[i] - bar_width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(collusive_vals):
        ax.text(x[i] + bar_width/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    # Add horizontal dashed grid lines
    ax.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered_models, rotation=20)
    ax.set_ylabel(bar_ylabels[idx], fontweight='bold')
    ax.set_title(bar_titles[idx], fontweight='bold')

# Add organized legend below all axes, two rows (one for each condition per model)
fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.18), fontsize=12, frameon=False)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig('barplots_collusive_vs_noncollusive.png', dpi=300, bbox_inches='tight')
print('Bar plots saved as barplots_collusive_vs_noncollusive.png')

# --- Alliance Score Heatmaps ---
def map_trial_models(models):
    if isinstance(models, str):
        try:
            models = ast.literal_eval(models)
        except Exception:
            return []
    return [MODEL_NAME_MAP.get(m, m) for m in models]

def compute_alliance_score_matrix(trial_df, trial_numbers):
    matrix = np.zeros((len(ordered_models), len(ordered_models)))
    counts = np.zeros((len(ordered_models), len(ordered_models)))
    for _, row in trial_df[trial_df['unique_trial_id'].isin(trial_numbers)].iterrows():
        scores = row['final_alliance_scores']
        if isinstance(scores, str):
            try:
                scores = ast.literal_eval(scores)
            except Exception:
                continue
        if not isinstance(scores, dict):
            continue
        trial_models = map_trial_models(row['models'])
        for i, m1 in enumerate(trial_models):
            for j, m2 in enumerate(trial_models):
                if i != j and m1 in ordered_models and m2 in ordered_models:
                    score = scores.get(str(i), {}).get(str(j), 0)
                    i1 = ordered_models.index(m1)
                    i2 = ordered_models.index(m2)
                    matrix[i1, i2] += score
                    counts[i1, i2] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_matrix = np.where(counts > 0, matrix / counts, 0)
    return avg_matrix

collusive_alliance_matrix = compute_alliance_score_matrix(trial_df, collusive_trial_ids)

# For the heatmap, use short names with model numbers
heatmap_labels = ['DeepSeek-R1', 'Claude-3.7', 'Gemini-2.0', 'o4-mini']

fig, ax = plt.subplots(figsize=(7, 6))
# Draw the heatmap
hm = sns.heatmap(
    collusive_alliance_matrix,
    annot=True,
    fmt='.2f',
    cmap='Purples',
    linewidths=0,  # No grid cell borders
    linecolor=None,
    cbar=True,
    ax=ax,
    square=True,
    annot_kws={"fontsize": 12}
)
ax.set_xticks(np.arange(len(heatmap_labels)) + 0.5)
ax.set_yticks(np.arange(len(heatmap_labels)) + 0.5)
ax.set_xticklabels(heatmap_labels)
ax.set_yticklabels(heatmap_labels)
ax.set_xlabel('Partner Model', fontweight='bold', fontsize=14)
ax.set_ylabel('Model', fontweight='bold', fontsize=14)
ax.set_title('Average Alliance Scores Between Models', fontweight='bold', fontsize=15)

# Add a yellow border around all grid squares with the highest value
max_val = np.nanmax(collusive_alliance_matrix)
max_indices = np.argwhere(collusive_alliance_matrix == max_val)
for idx in max_indices:
    rect = plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='yellow', linewidth=3)
    ax.add_patch(rect)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('alliance_score_heatmap_collusive.png', dpi=300)
print('Alliance score heatmap (collusive only) saved as alliance_score_heatmap_collusive.png')

# --- Additional: Collusive Victories, Signals Sent, Alliances Formed (Collusive Only) ---
model_colors = {
    'DeepSeek-R1': '#8F7289',
    'Claude-3.7': '#ED7B61',
    'Gemini-2.0': '#29A15C',
    'o4-mini': '#529AC9'
}

# For the collusive grouped bar plot, use only the last 5 sets of 4 rows (collusive trials)
collusive_trial_set = set(collusive_trial_ids)

# Map long to short names for the bar labels
model_long_to_short = {
    'DeepSeek-R1': 'DeepSeek-R1',
    'Claude-3.7-Sonnet': 'Claude-3.7',
    'Gemini-2.0-Flash': 'Gemini-2.0',
    'o4-mini': 'o4-mini'
}
collusive_bar_labels = [model_long_to_short[m] for m in ordered_models]

# Alliances formed: count for each model in collusive trials based on final alliance scores > 2
alliances_formed_counts = {label: 0 for label in collusive_bar_labels}
for _, row in collusive_trial_rows.iterrows():
    # Parse models and final_alliance_scores
    models = row['models']
    if isinstance(models, str):
        try:
            models = ast.literal_eval(models)
        except Exception:
            continue
    model_names = [MODEL_NAME_MAP.get(m, m) for m in models]
    final_scores = row['final_alliance_scores']
    if isinstance(final_scores, str):
        try:
            final_scores = ast.literal_eval(final_scores)
        except Exception:
            continue
    # Count alliances with score > 2
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i != j:
                score = final_scores.get(str(i), {}).get(str(j), 0)
                if score > 2:
                    alliances_formed_counts[model_long_to_short.get(m1, m1)] += 1
collusive_alliances_formed = [alliances_formed_counts[label] for label in collusive_bar_labels]

# Victories: count unique winners per trial (collusive only)
winner_counts = {label: 0 for label in collusive_bar_labels}
print('Debug: Collusive trial winner mapping:')
for trial in collusive_trial_ids:
    winners = collusive_trial_rows[collusive_trial_rows['unique_trial_id'] == trial]['winner']
    if not winners.empty:
        winner_list = ast.literal_eval(str(winners.values[0]))
        for agent_id in winner_list:
            agent_row = collusive_agent_rows[(collusive_agent_rows['unique_trial_id'] == trial) & (collusive_agent_rows['agent_id'].astype(int) == int(agent_id))]
            if not agent_row.empty:
                model_long = agent_row.iloc[0]['model']
                model_short = model_long_to_short.get(model_long, model_long)
                print(f'Trial {trial}, Winner agent_id: {agent_id}, Model long: {model_long}, Model short: {model_short}')
                if model_short in winner_counts:
                    winner_counts[model_short] += 1
collusive_victories = [winner_counts[label] for label in collusive_bar_labels]

# Print victories per model and total victories for collusive trials
print('Victories per model (collusive trials):')
for label in collusive_bar_labels:
    print(f'{label}: {winner_counts[label]}')
print('Total victories:', sum(winner_counts.values()))

# Signals sent: sum for each model in collusive trials
signals_sent_counts = {label: 0 for label in collusive_bar_labels}
for label, long_name in zip(collusive_bar_labels, ordered_models):
    signals_sent_counts[label] = collusive_agent_rows[collusive_agent_rows['model'] == long_name]['alliance_signals_sent'].sum()
collusive_signals_sent = [signals_sent_counts[label] for label in collusive_bar_labels]

# River cleaning: sum for each model in collusive trials
river_cleaning_counts = {label: 0 for label in collusive_bar_labels}
for label, long_name in zip(collusive_bar_labels, ordered_models):
    river_cleaning_counts[label] = collusive_agent_rows[collusive_agent_rows['model'] == long_name]['river_cleaning_actions'].sum()
collusive_river_cleaning = [river_cleaning_counts[label] for label in collusive_bar_labels]

# --- Combined: Collusive Grouped Bar Plot and Alliance Score Heatmap Side by Side ---
# Use a larger bar width and tighter x spacing for uniform plot sizes
combined_bar_width = 0.28
x_combined = np.arange(len(collusive_bar_labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 1]})

# --- Collusive Grouped Bar Plot (ax1) ---
ax1.bar(x_combined - combined_bar_width, collusive_victories, width=combined_bar_width, label='Victories', color=[model_colors[l] for l in collusive_bar_labels], alpha=0.9)
ax1.bar(x_combined, collusive_signals_sent, width=combined_bar_width, label='Signals Sent', color=[model_colors[l] for l in collusive_bar_labels], alpha=0.5)
ax1.bar(x_combined + combined_bar_width, collusive_alliances_formed, width=combined_bar_width, label='Alliances Formed', color=[model_colors[l] for l in collusive_bar_labels], alpha=0.3)
# Add value labels
for i, v in enumerate(collusive_victories):
    ax1.text(x_combined[i] - combined_bar_width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(collusive_signals_sent):
    ax1.text(x_combined[i], v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(collusive_alliances_formed):
    ax1.text(x_combined[i] + combined_bar_width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
ax1.yaxis.grid(True, linestyle='--', linewidth=1, alpha=0.5)
ax1.set_axisbelow(True)
ax1.set_xticks(x_combined)
ax1.set_xticklabels(collusive_bar_labels, rotation=0)
ax1.set_ylabel('Count', fontweight='bold', fontsize=14)
ax1.set_title('Collusive Metrics by Model', fontweight='bold')
ax1.legend(loc='upper right')
ax1.set_xlim(-0.7, len(collusive_bar_labels)-0.3)
ax1.set_aspect('auto')

# --- Alliance Score Heatmap (ax2) ---
hm = sns.heatmap(
    collusive_alliance_matrix,
    annot=True,
    fmt='.2f',
    cmap='Purples',
    linewidths=0,
    linecolor=None,
    cbar=True,
    ax=ax2,
    square=True,
    annot_kws={"fontsize": 12}
)
ax2.set_xticks(np.arange(len(heatmap_labels)) + 0.5)
ax2.set_yticks(np.arange(len(heatmap_labels)) + 0.5)
ax2.set_xticklabels(heatmap_labels, rotation=0)
ax2.set_yticklabels(heatmap_labels)
ax2.set_xlabel('Partner Model', fontweight='bold', fontsize=14)
ax2.set_ylabel('Model', fontweight='bold', fontsize=14)
ax2.set_title('Average Alliance Scores Between Models', fontweight='bold', fontsize=15)
# Add a yellow border around all grid squares with the highest value
max_val = np.nanmax(collusive_alliance_matrix)
max_indices = np.argwhere(collusive_alliance_matrix == max_val)
for idx in max_indices:
    rect = plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='yellow', linewidth=3)
    ax2.add_patch(rect)

plt.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('collusive_bar_and_heatmap.png', dpi=300)
print('Combined collusive bar plot and heatmap saved as collusive_bar_and_heatmap.png') 