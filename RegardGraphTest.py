import pandas as pd
import ternary
import matplotlib.pyplot as plt

def plot_regard_ternary(data):
    # If a file path is provided, read the CSV
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Use scale=100 since the scores are in percentages
    scale = 100

    # Create a bigger figure
    figure, tax = ternary.figure(scale=scale)
    figure.set_size_inches(8, 7)  # Image size (width, height)

    # Set title and styling
    tax.set_title("Ternary Plot of Regard Scores by Type", fontsize=18, pad=20)
    tax.boundary(linewidth=1.5)
    tax.gridlines(multiple=10, color="gray", linewidth=0.5)

    # Axis labels with your desired orientation
    fontsize = 12
    tax.left_axis_label("Regard Negative", fontsize=fontsize, offset=0.14)
    tax.right_axis_label("Regard Positive", fontsize=fontsize, offset=0.14)
    tax.bottom_axis_label("Regard Neutral", fontsize=fontsize, offset=0.10)

    # Set tick marks
    tax.ticks(axis='lbr', linewidth=1, multiple=10, fontsize=10)

    # Style map for each type
    type_styles = {
        "Queer": {"color": "red", "marker": "o"},
        "Non Queer": {"color": "green", "marker": "o"},
        "Unmarked": {"color": "orange", "marker": "o"},
    }

    # Plot points with orientation: (left, right, bottom) => (Negative, Positive, Neutral)
    for t_type, style in type_styles.items():
        df_filtered = data[data["type"] == t_type]
        points = list(zip(
            df_filtered["Regard Negative"],  # left
            df_filtered["Regard Positive"],  # right
            df_filtered["Regard Neutral"]    # bottom
        ))
        tax.scatter(
            points,
            label=t_type,
            color=style["color"],
            marker=style["marker"],
            s=10,           # smaller dots
            alpha=0.7       # transparency
        )

    # Legend and final cleanup
    tax.legend(fontsize=11, loc='upper right')
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    tax.show()
    tax.savefig('test/test.png', bbox_inches = 'tight')
    
def process_regard_scores(input_file, output_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Extract relevant columns and multiply Regard scores by 100
    result_df = df[['type', 'Regard Positive', 'Regard Neutral', 'Regard Negative']].copy()
    #result_df = result_df.sort_values(by='type')
    result_df['Regard Positive'] = round(result_df['Regard Positive'] * 100, 1)
    result_df['Regard Neutral'] = round(result_df['Regard Neutral'] * 100, 1)
    result_df['Regard Negative'] = round(result_df['Regard Negative'] * 100, 1)

    # Save the result to a new CSV file
    result_df.to_csv(output_file, index=False)
    plot_regard_ternary(result_df)

# Usage example
input_file = 'output_evaluation/gpt-4o.csv'  # Replace with the path to your input CSV file
output_file = 'test/gpt-4o.csv'  # Replace with the desired output file path
process_regard_scores(input_file, output_file)