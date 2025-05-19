import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-3-7-sonnet-20250219",
    max_tokens=2048,
    tools=[
        {
            "name": "terminal",
            "description": "Pipe input to the terminal (tty) stdin. Console output will be returned after a maximum of 3 seconds, although the command may still be running after output. You can wait for it to finish and check-in on progress or stop it by doing another tool call. Examples: <terminal>ls -la\n</terminal> or <terminal>^C</terminal>.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "String to be piped into the terminal (tty) stdin. Tip: this is a generic stdin input, so if you want to execute a full command, you need to emit a newline after the command. Also, you can type arbitrary control codes like ^C, ^D, ^[, ^[OP, ^[[A, etc.",
                    }
                },
                "required": ["input"],
            },
        }
    ],
    messages=[{"role": "user", "content": "What's the current python version?"}],
    temperature=1.0,
)
print(response)
