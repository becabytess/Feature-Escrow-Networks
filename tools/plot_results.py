import os
import matplotlib.pyplot as plt
import numpy as np

# Create the results directory
os.makedirs("results", exist_ok=True)

# Set custom styling parameters for clean, premium academic look
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 0.8,
    'grid.color': '#eeeeee',
    'grid.linewidth': 0.5,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

# Color palette
colors = {
    'RNN': '#8E9AAF',        # Slate gray
    'LSTM': '#FF70A6',       # Warm pink
    'PDN': '#2E86AB',        # Deep blue
    'PDN_MLP': '#F26419'     # Orange/coral
}

# -------------------------------------------------------------
# Plot 1: Synthetic stress tests (Accuracy %)
# -------------------------------------------------------------
scenarios = [
    'Distracted Count\n(Beginning)', 
    'Distracted Count\n(Random)', 
    'Delayed Recall\n(Beginning)', 
    'Delayed Recall\n(Random)'
]
rnn_acc = [10, 66, 10, 10]
lstm_acc = [10, 100, 10, 10]
pdn_acc = [100, 100, 91, 10]

x = np.arange(len(scenarios))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.grid(axis='y', linestyle='--', alpha=0.7)

rects1 = ax.bar(x - width, rnn_acc, width, label='Raw RNN', color=colors['RNN'], edgecolor='none', alpha=0.95)
rects2 = ax.bar(x, lstm_acc, width, label='LSTM', color=colors['LSTM'], edgecolor='none', alpha=0.95)
rects3 = ax.bar(x + width, pdn_acc, width, label='PDN (Ours)', color=colors['PDN'], edgecolor='none', alpha=0.95)

ax.set_ylabel('Peak Validation Accuracy (%)', fontweight='bold')
ax.set_title('Synthetic Stress-Tests: Memory & Interference', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylim(0, 115)
ax.legend(frameon=True, facecolor='white', edgecolor='none')

# Add values above bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, color='#444444')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.savefig('results/synthetic_results.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Plot 2: UCI HAR Biomechanics (Accuracy & Parameter Efficiency)
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)

# Accuracy chart
labels = ['Metadata-Assisted\n(Subject ID given)', 'Blind\n(No Subject ID)']
lstm_har = [87.8, 87.8]
pdn_har = [89.7, 92.1]

x = np.arange(len(labels))
width = 0.35

ax1.grid(axis='y', linestyle='--', alpha=0.7)
r1 = ax1.bar(x - width/2, lstm_har, width, label='LSTM', color=colors['LSTM'], alpha=0.95)
r2 = ax1.bar(x + width/2, pdn_har, width, label='PDN (Ours)', color=colors['PDN'], alpha=0.95)
ax1.set_ylabel('Peak Test Accuracy (%)', fontweight='bold')
ax1.set_title('Classification Performance', fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(80, 95)
ax1.legend(frameon=True, facecolor='white', edgecolor='none')

for rect in r1:
    ax1.annotate(f'{rect.get_height()}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in r2:
    ax1.annotate(f'{rect.get_height()}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

# Parameters chart
lstm_params = [32756, 12506]
pdn_params = [10256, 12306]

ax2.grid(axis='y', linestyle='--', alpha=0.7)
p1 = ax2.bar(x - width/2, lstm_params, width, label='LSTM', color=colors['LSTM'], alpha=0.7)
p2 = ax2.bar(x + width/2, pdn_params, width, label='PDN (Ours)', color=colors['PDN'], alpha=0.95)
ax2.set_ylabel('Parameter Count', fontweight='bold')
ax2.set_title('Model Size (Parameter Efficiency)', fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'{val/1000:.1f}k'))

for rect in p1:
    ax2.annotate(f'{rect.get_height():,}', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in p2:
    ax2.annotate(f'{rect.get_height():,}', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.suptitle('UCI HAR Dataset: Activity Recognition Benchmarks', fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('results/uci_har_results.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Plot 3: Clinical ICU Prediction (PhysioNet Mortality Recall vs Acc)
# -------------------------------------------------------------
models = ['Raw RNN', 'LSTM', 'PDN (Ours)']
mort_recall = [58.1, 57.3, 82.1]
avg_recall = [48.0, 52.0, 68.0]
test_acc = [65.0, 69.0, 69.0]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
ax.grid(axis='y', linestyle='--', alpha=0.7)

b1 = ax.bar(x - width, mort_recall, width, label='Peak Mortality Recall (Critical)', color='#D90429', alpha=0.95)
b2 = ax.bar(x, avg_recall, width, label='Average Recall', color='#F77F00', alpha=0.9)
b3 = ax.bar(x + width, test_acc, width, label='Test Accuracy', color='#4A90E2', alpha=0.85)

ax.set_ylabel('Percentage (%)', fontweight='bold')
ax.set_title('PhysioNet ICU 2012: In-Hospital Mortality Prediction', fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 100)
ax.legend(frameon=True, facecolor='white', edgecolor='none')

for rect in b1:
    ax.annotate(f'{rect.get_height()}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
for rect in b2:
    ax.annotate(f'{rect.get_height()}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
for rect in b3:
    ax.annotate(f'{rect.get_height()}%', xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/physionet_mortality_results.png', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------
# Plot 4: ICU Length of Stay (Recall Stability)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Bar chart showing Peak vs Final recall for Long Stay class
categories = ['LSTM (Peak)', 'LSTM (Final)', 'PDN (Peak)', 'PDN (Final)']
recall_vals = [74, 59, 76, 68]
bar_colors = [colors['LSTM'], '#FFB5D0', colors['PDN'], '#7FBDE0']

bars = ax.bar(categories, recall_vals, color=bar_colors, width=0.5)
ax.set_ylabel('Long-Stay Recall (%)', fontweight='bold')
ax.set_title('Clinical Context Drift: Long-Stay Recall Stability\n(Overfitting Protection / Late-Stage Stability)', fontweight='bold', pad=15)
ax.set_ylim(0, 90)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add arrow indicating change
# LSTM drop
ax.annotate('', xy=(0.8, 60), xytext=(0.2, 73),
            arrowprops=dict(facecolor='red', shrink=0.08, width=1.5, headwidth=6))
ax.text(0.5, 68, '-15%', color='red', ha='center', fontweight='bold')

# PDN drop
ax.annotate('', xy=(2.8, 69), xytext=(2.2, 75),
            arrowprops=dict(facecolor='orange', shrink=0.08, width=1.5, headwidth=6))
ax.text(2.5, 73, '-8%', color='orange', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('results/physionet_los_results.png', bbox_inches='tight')
plt.close()

print("All plots successfully generated and saved to results/ folder.")
