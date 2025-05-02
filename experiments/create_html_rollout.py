import html
import json
import sys


def create_html_from_jsonl(input_file, output_file):
    """
    Read a JSONL file and create an HTML file displaying the input and output.

    Args:
        input_file: Path to the JSONL file
        output_file: Path to save the HTML output
    """
    with open(input_file, "r", encoding="utf-8") as f:
        html_content = []

        # Start HTML document
        html_content.append("""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JSONL Data Visualization</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    text-align: center;
                    margin-bottom: 30px;
                }
                .sample {
                    background-color: #f9f9f9;
                    border-radius: 8px;
                    padding: 20px;
                    margin-bottom: 30px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .sample-header {
                    display: flex;
                    justify-content: space-between;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                    margin-bottom: 15px;
                }
                .section {
                    margin-bottom: 20px;
                }
                .section-title {
                    font-weight: bold;
                    color: #3498db;
                    margin-bottom: 10px;
                }
                .code-block {
                    background-color: #f0f0f0;
                    padding: 15px;
                    border-radius: 4px;
                    font-family: monospace;
                    white-space: pre-wrap;
                    overflow-x: auto;
                }
                .think-block {
                    background-color: #e8f4f8;
                    padding: 10px;
                    border-left: 3px solid #3498db;
                    margin-bottom: 10px;
                }
                .terminal-block {
                    background-color: #2c3e50;
                    color: #ecf0f1;
                    padding: 10px;
                    border-radius: 4px;
                    font-family: monospace;
                    white-space: pre-wrap;
                }
                .output-block {
                    background-color: #f5f5f5;
                    padding: 10px;
                    border-left: 3px solid #7f8c8d;
                    font-family: monospace;
                    white-space: pre-wrap;
                }
                .answer-block {
                    background-color: #e8f8e8;
                    padding: 10px;
                    border-left: 3px solid #27ae60;
                    margin-bottom: 10px;
                }
                .metadata {
                    font-size: 0.8em;
                    color: #7f8c8d;
                }
            </style>
        </head>
        <body>
            <h1>JSONL Data Visualization</h1>
        """)

        # Process each line of the JSONL file
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())

                # Extract data
                input_data = data.get("input", "")
                output_data = data.get("output", "")
                score = data.get("score", "N/A")
                step = data.get("step", "N/A")

                # Split input into system and user parts
                if "system" in input_data and "user" in input_data:
                    # This assumes the format is "system\n{system_content}\nuser\n{user_content}"
                    parts = input_data.split("user\n", 1)
                    system_content = (
                        parts[0].replace("system\n", "", 1) if len(parts) > 0 else ""
                    )
                    user_content = parts[1] if len(parts) > 1 else ""
                else:
                    system_content = input_data
                    user_content = ""

                # Format the output data - process tags like <think>, <terminal>, etc.
                formatted_output = output_data
                # Replace <think> blocks
                formatted_output = formatted_output.replace(
                    "<think>", '<div class="think-block"><strong>Think:</strong> '
                )
                formatted_output = formatted_output.replace("</think>", "</div>")
                # Replace <terminal> blocks
                formatted_output = formatted_output.replace(
                    "<terminal>",
                    '<div class="terminal-block"><strong>Terminal:</strong>\n',
                )
                formatted_output = formatted_output.replace("</terminal>", "</div>")
                # Replace <output> blocks
                formatted_output = formatted_output.replace(
                    "<output>", '<div class="output-block"><strong>Output:</strong>\n'
                )
                formatted_output = formatted_output.replace("</output>", "</div>")
                # Replace <answer> blocks
                formatted_output = formatted_output.replace(
                    "<answer>", '<div class="answer-block"><strong>Answer:</strong> '
                )
                formatted_output = formatted_output.replace("</answer>", "</div>")

                # Create HTML for this sample
                html_content.append(f"""
                <div class="sample" id="sample-{i + 1}">
                    <div class="sample-header">
                        <h2>Sample {i + 1}</h2>
                        <div class="metadata">
                            <span>Score: {score}</span> | <span>Step: {step}</span>
                        </div>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">System Input:</div>
                        <div class="code-block">{html.escape(system_content)}</div>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">User Input:</div>
                        <div class="code-block">{html.escape(user_content)}</div>
                    </div>
                    
                    <div class="section">
                        <div class="section-title">Output:</div>
                        <div>{formatted_output}</div>
                    </div>
                </div>
                """)

            except json.JSONDecodeError:
                html_content.append(f"""
                <div class="error">
                    <p>Error parsing line {i + 1}. Invalid JSON.</p>
                    <pre>{html.escape(line)}</pre>
                </div>
                """)

        # Close HTML document
        html_content.append("""
        </body>
        </html>
        """)

        # Write the HTML file
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(html_content))


if __name__ == "__main__":
    if len(sys.argv) > 2:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = input("Enter the path to your JSONL file: ")
        output_file = input("Enter the path for the HTML output file: ")

    create_html_from_jsonl(input_file, output_file)
    print(f"HTML file created at: {output_file}")
