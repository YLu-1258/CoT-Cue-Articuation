import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd

def load_evaluations(file_path):
    evaluations = []
    with open(file_path, 'r') as f:
        for line in f:
            evaluations.append(json.loads(line))
    return evaluations

def analyze_cue_articulation(evaluations):
    # Count acknowledged vs not acknowledged cues
    cue_counts = Counter(eval['acknowledged_cue'] for eval in evaluations)
    
    # Calculate percentages
    total = len(evaluations)
    acknowledged_pct = (cue_counts['yes'] / total) * 100
    not_acknowledged_pct = (cue_counts['no'] / total) * 100
    
    return {
        'acknowledged': acknowledged_pct,
        'not_acknowledged': not_acknowledged_pct,
        'raw_counts': cue_counts
    }

def create_visualizations():
    # Load both evaluation files
    fewshot_evaluations = load_evaluations('fewshot_black_squares_evaluations.jsonl')
    stanford_evaluations = load_evaluations('stanford_professor_evaluations.jsonl')
    
    # Analyze cue articulation for both files
    fewshot_analysis = analyze_cue_articulation(fewshot_evaluations)
    stanford_analysis = analyze_cue_articulation(stanford_evaluations)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for fewshot black squares
    labels = ['Acknowledged', 'Not Acknowledged']
    sizes = [fewshot_analysis['acknowledged'], fewshot_analysis['not_acknowledged']]
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Fewshot Black Squares Cue Articulation')
    
    # Plot for stanford professor
    sizes = [stanford_analysis['acknowledged'], stanford_analysis['not_acknowledged']]
    ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Stanford Professor Cue Articulation')
    
    plt.tight_layout()
    plt.savefig('cue_articulation_rates.png')
    plt.close()
    
    # Create a bar plot comparing both
    df = pd.DataFrame({
        'Cue Type': ['Fewshot Black Squares', 'Stanford Professor'] * 2,
        'Status': ['Acknowledged', 'Acknowledged', 'Not Acknowledged', 'Not Acknowledged'],
        'Percentage': [
            fewshot_analysis['acknowledged'],
            stanford_analysis['acknowledged'],
            fewshot_analysis['not_acknowledged'],
            stanford_analysis['not_acknowledged']
        ]
    })
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Cue Type', y='Percentage', hue='Status')
    plt.title('Comparison of Cue Articulation Rates')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('cue_articulation_comparison.png')
    plt.close()
    
    # Print raw statistics
    print("\nFewshot Black Squares Statistics:")
    print(f"Total evaluations: {len(fewshot_evaluations)}")
    print(f"Raw counts: {fewshot_analysis['raw_counts']}")
    print(f"Acknowledged: {fewshot_analysis['acknowledged']:.1f}%")
    print(f"Not acknowledged: {fewshot_analysis['not_acknowledged']:.1f}%")
    
    print("\nStanford Professor Statistics:")
    print(f"Total evaluations: {len(stanford_evaluations)}")
    print(f"Raw counts: {stanford_analysis['raw_counts']}")
    print(f"Acknowledged: {stanford_analysis['acknowledged']:.1f}%")
    print(f"Not acknowledged: {stanford_analysis['not_acknowledged']:.1f}%")

if __name__ == "__main__":
    create_visualizations() 