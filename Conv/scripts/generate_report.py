#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def generate_performance_report(csv_file, output_file):
    df = pd.read_csv(csv_file)
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
    	<meta charset="UTF-8">
        <title>DCU Convolution Performance Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .table {{ border-collapse: collapse; width: 100%; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
            .table th {{ background-color: #f2f2f2; }}
            .highlight {{ background-color: #e8f5e8; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>DCU Convolution Performance Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Test Platform: 异构加速卡1 (16GB显存)</p>
        </div>

        <div class="section">
            <h2>Test Summary</h2>
            <p>Total test cases: {len(df)}</p>
            <p>Average GPU Basic speedup: {df['speedup_basic'].mean():.2f}x</p>
            <p>Average GPU Optimized speedup: {df['speedup_optimized'].mean():.2f}x</p>
            <p>Average GPU Tiled speedup: {df['speedup_tiled'].mean():.2f}x</p>
        </div>

        <div class="section">
            <h2>Detailed Results</h2>
            <table class="table">
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Input Size</th>
                        <th>CPU Time (ms)</th>
                        <th>GPU Basic (ms)</th>
                        <th>GPU Optimized (ms)</th>
                        <th>GPU Tiled (ms)</th>
                        <th>Best Speedup</th>
                    </tr>
                </thead>
                <tbody>
    """

    for idx, row in df.iterrows():
        input_size = f"{row['batch_size']}×{row['input_channels']}×{row['input_height']}×{row['input_width']}"
        best_speedup = max(row['speedup_basic'], row['speedup_optimized'], row['speedup_tiled'])
        html_content += f"""
                    <tr>
                        <td>Case {idx+1}</td>
                        <td>{input_size}</td>
                        <td>{row['cpu_time']:.2f}</td>
                        <td>{row['gpu_basic_time']:.2f}</td>
                        <td>{row['gpu_optimized_time']:.2f}</td>
                        <td>{row['gpu_tiled_time']:.2f}</td>
                        <td class="highlight">{best_speedup:.2f}x</td>
                    </tr>
        """

    html_content += """
                </tbody>
            </table>
        </div>
    """

    # 性能分析部分
    best_idx = df['speedup_tiled'].idxmax()
    best_case = df.loc[best_idx]
    avg_improvement = (df['gpu_basic_time'] / df['gpu_tiled_time']).mean()

    html_content += f"""
        <div class="section">
            <h2>Performance Analysis</h2>
            <h3>Key Findings:</h3>
            <ul>
                <li>Best performance achieved in Case {best_idx+1} with {best_case['speedup_tiled']:.2f}x speedup</li>
                <li>Tiled implementation shows average {avg_improvement:.2f}x improvement over basic GPU implementation</li>
                <li>Memory optimization is crucial for large-scale convolutions</li>
            </ul>
        </div>
    </body>
    </html>
    """

    with open(output_file, 'w') as f:
        f.write(html_content)

    print(f"Performance report generated: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 generate_report.py <csv_file> <output_html>")
        sys.exit(1)

    csv_file = sys.argv[1]
    output_file = sys.argv[2]

    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        sys.exit(1)

    generate_performance_report(csv_file, output_file)
