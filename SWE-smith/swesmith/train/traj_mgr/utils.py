import json
import yaml

XML_STR_REPLACES = ["old_str", "new_str", "file_text"]


# TODO: Fix this, this is hardcoded, so will break if not called from root of a directory
SYSTEM_PROMPT = yaml.safe_load(open("agent/swesmith_infer.yaml", "r"))["agent"][
    "templates"
]["system_template"]


def transform_traj_backticks(traj: dict) -> dict:
    new_traj = []
    for message in traj["trajectory"][-1]["messages"][:-1]:
        # Pick out the last message b/c it contains full trajectory
        # Also, skip the last message b/c it's just the patch output (post-submit)
        role = message["role"] if message["role"] != "tool" else "user"
        if message["role"] == "assistant":
            content = f"{message['thought']}\n\n```\n{message['action']}\n```"
        elif message["role"] == "system":
            content = message["content"]
        else:
            assert len(message["content"]) == 1
            content = message["content"][0]["text"]
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}


def transform_traj_xml(traj: dict) -> dict:
    def tool_call_to_action(tool_calls):
        actions = []
        if tool_calls is None:
            return []
        for tool_call in tool_calls:
            action = [f"<function={tool_call['function']['name']}>"]
            arguments = json.loads(tool_call["function"]["arguments"])
            for k, v in arguments.items():
                a = f"<parameter={k}>{v}</parameter>"
                if k in XML_STR_REPLACES:
                    a = f"<parameter={k}>\n{v}\n</parameter>"
                action.append(a)
            action.append("</function>")
            actions.append("\n".join(action))
        return actions

    new_traj = []
    messages = traj["trajectory"][-1]["messages"][:-1]
    for message in messages:
        role = message["role"] if message["role"] != "tool" else "user"
        if message["role"] == "assistant":
            if message["content"] == "Exit due to cost limit":
                content = (
                    "Since we have successfully fixed the issue and verified it works, "
                    + "let's submit the changes:\n\n"
                    + "<function=submit>\n</function>"
                )
            else:
                action = "\n".join(tool_call_to_action(message["tool_calls"]))
                content = f"{message['thought']}\n\n{action}"
        elif message["role"] == "system":
            content = SYSTEM_PROMPT
        else:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                content = message["content"][0]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            else:
                raise ValueError(f"Message type not recognized: {type(message)}")
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}


MAP_STYLE_TO_FUNC = {"ticks": transform_traj_backticks, "xml": transform_traj_xml}
