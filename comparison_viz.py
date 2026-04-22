# comparison_viz.py
import matplotlib.pyplot as plt
import numpy as np

def create_comparison_chart(cc_conf, mlo_conf):
    """Create a bar chart comparing view confidences"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    views = ['CC View', 'MLO View']
    confidences = [cc_conf, mlo_conf]
    colors = ['#ff6b6b' if c > 0.5 else '#4ecdc4' for c in confidences]
    
    bars = ax.bar(views, confidences, color=colors)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Confidence')
    ax.set_title('View Confidence Comparison')
    
    # Add value labels on bars
    for bar, conf in zip(bars, confidences):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{conf:.1%}', ha='center', va='bottom')
    
    return fig

def create_agreement_heatmap(cc_pred, mlo_pred):
    """Create a simple agreement visualization"""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    data = np.array([[1 if cc_pred == mlo_pred else 0]])
    ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('View Agreement\n🟢 Agree 🔴 Disagree')
    
    for i in range(1):
        for j in range(1):
            text = '✓' if cc_pred == mlo_pred else '✗'
            ax.text(j, i, text, ha='center', va='center', 
                   fontsize=20, color='white')
    
    return fig