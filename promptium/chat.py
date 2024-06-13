
import ezpyz as ez
import pyperclip as pc # noqa
from promptium.prompt import prompt
import re
import uuid
from promptium.gpt import GPT, OpenAiAccount

account = OpenAiAccount()


gpt = GPT(account, 'gpt-4')

instructions = """
Interactive GPT chat for terminal.

Commands:
    ? - Send current user message
    U-[n] - Use previous user message
    S-[n] - Use previous system message
    clear - Clear/start over the current user message
    copy - Copy previous system message to clipboard
    exit - Exit chat
    <anything else> - Append line to current user message
"""


def main():
    print(instructions)
    session_name = input('Enter a session name: ')
    if not session_name.strip():
        session_name = str(uuid.uuid4())
    session = ez.Store[list[tuple[str, str]]](path=f'chats/{session_name}.json')
    if session.path.exists():
        session.load()
        print('\n'.join(f'{user_msg}\n{system_msg}\n' for user_msg, system_msg in session.data))
        print()
    else:
        session.data = []
    while True:
        context = []
        user_lines = []
        while (user_line := input()) != '?':
            if user_line.strip() == 'clear':
                user_lines.clear()
                print('Cleared user message, start over:', '\n')
            elif matches:=re.fullmatch(r'U-[0-9]+', user_line.strip()):
                back_index = int(matches[0][2:])
                if back_index > len(user_lines):
                    print('No such line')
                    continue
                previous_user_turn = session.data[-back_index][0]
                context.append(previous_user_turn)
                print(f"U: {previous_user_turn[:50]}...")
            elif matches:=re.fullmatch(r'S-[0-9]', user_line.strip()):
                back_index = int(matches[0][2:])
                if back_index > len(session.data):
                    print('No such line')
                    continue
                previous_system_turn = session.data[-back_index][1]
                context.append(previous_system_turn)
                print(f"S: {previous_system_turn[:50]}...")
            elif user_line.strip() == 'exit':
                return
            elif user_line.strip() == 'copy' and len(session.data) > 0:
                pc.copy(session.data[-1][1])
                print(f"Copied {session.data[-1][1][:50]}...")
            else:
                user_lines.append(user_line)
        user_msg = '\n'.join(user_lines)
        call = GPT(account, 'gpt-4',
            history=context,
        )
        system_msg = call(user_msg)
        print(system_msg, '\n')
        session.data.append((user_msg, system_msg))



if __name__ == '__main__':
    main()
