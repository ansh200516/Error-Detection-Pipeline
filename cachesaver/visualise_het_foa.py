import re
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
import colorsys
from PIL import Image, ImageDraw, ImageFont
import random
import os
from termcolor import colored

# these will be given by the user
logs = ''
state_names = {}
states_done_in_puzzle = {}
state_colors = {}

class State(BaseModel):
    name: str
    color: str
    num_thoughts: int
    serial_data: dict
    value: Optional[float] = None
    terminal_data: str = ''


class Timestep(BaseModel):
    timestep: int
    input_states: list[State]
    agent_output_states: list[State]
    state_wins: list[bool]
    state_fails: list[bool]
    replacement_states: list[State]
    values: Optional[list[float]] = None



def generate_distinct_hex_colors(n):
    """
    Generate `n` distinct hex colors that are as different as possible and not close to black.
    
    Returns:
        List of hex color strings (e.g., '#FF5733').
    """
    colors = []
    for i in range(n):
        # Evenly space hues around the color wheel
        hue = i / n
        saturation = 0.65  # Keep saturation high to avoid washed-out colors
        value = 0.8        # Avoid dark (black-ish) colors by setting high brightness
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors


def load_logs() -> str:
    with open(f"logs/het_foa_logs.log", 'r') as f:
        logs = f.read()
        logs = logs.split('#################################################################')[-1].strip()
    return logs


def get_puzzle_idx(log):
    res = re.search(r"het_foa_logs-(\d+)-", log)
    assert res is not None, f'Puzzle index not found in log: {log}'
    return int(res.group(1))


def get_timestep(log):
    res = re.search(r"het_foa_logs-(\d+)-(\d+)", log)
    assert res is not None, f'Timestep not found in log: {log}'
    return int(res.group(1))


def get_py_list(string, type):
    l = eval(string)
    assert isinstance(l, list), f'Expected a list, got {type(l)}: {l}'

    for i, item in enumerate(l):
        l[i] = type(item)

    assert all(isinstance(item, type) for item in l), f'Expected all items to be {type.__name__}, got {l}'
    return l


def get_fleet(log):
    log = log.replace('ValueFunctionWrapped', '').replace('EnvWrapped', '')
    isolated_list = log.split('fleet: ')[-1].strip()
    return get_py_list(isolated_list, str)


def state_name(current_state: str, index):
    if hash(current_state) in state_names:
        return state_names[hash(current_state)]
    
    if index not in states_done_in_puzzle:
        states_done_in_puzzle[index] = 1
    states_done_in_puzzle[index] += 1
    
    idx = states_done_in_puzzle[index]
    idx = len(state_names)
    state_names[hash(current_state)] = f's{idx}'
    return state_names[hash(current_state)]


def get_state_color(state_name: str):
    if state_name in state_colors:
        return state_colors[state_name]
    
    idx = len(state_colors)
    state_colors[state_name] = f'color{idx}'
    return state_colors[state_name]


def get_states_from_log(log):
    index = get_puzzle_idx(log)
    isolated_list = log[log.find('['):]
    states = get_py_list(isolated_list, str)
    
    for i, state in enumerate(states):
        # load python dict from string
        if isinstance(state, str):
            try:
                states[i] = json.loads(state)
            except json.JSONDecodeError:
                raise ValueError(f'Invalid JSON in state: {state}')
    

    for i, state in enumerate(states):
        states[i] = State(
            name=state_name(state['current_state'], index),
            color=get_state_color(state_name(state['current_state'], index)),
            num_thoughts=len(state['reflections']),
            value=state['value'],
            serial_data=state
        )

    return states


def get_timestep_object(logs, timestep=0):
    # assert len(logs) == 6, f'Expected 6 logs for a timestep, got {len(logs)}: {logs}'

    assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
    if len(logs) > 3: assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
    if len(logs) > 4: assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'
    if len(logs) > 5: assert re.search(r'het_foa_logs-\d+-\d+-values', logs[5]), f'6th log does not match expected format: {logs[5]}'

    win_list = get_py_list(logs[2].split('statewins: ')[-1].strip(), bool)

    return Timestep(
        timestep=timestep,
        input_states=get_states_from_log(logs[0]),
        agent_output_states=get_states_from_log(logs[1]),
        state_wins=win_list,
        state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool) if len(logs) > 3 else [False] * len(win_list),
        replacement_states=get_states_from_log(logs[4]) if len(logs) > 4 else [],
        values=get_py_list(logs[5].split('values: ')[-1].strip(), float) if len(logs) > 5 else None
    )


def get_final_timestep(logs, timestep=0):
    assert len(logs) == 5, f'Expected 5 logs for a timestep, got {len(logs)}: {logs}'

    assert re.search(r'het_foa_logs-\d+-\d+-agentinputs', logs[0]), f'First log does not match expected format: {logs[0]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentouts', logs[1]), f'Second log does not match expected format: {logs[1]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statewins', logs[2]), f'Third log does not match expected format: {logs[2]}'
    assert re.search(r'het_foa_logs-\d+-\d+-statefails', logs[3]), f'4th log does not match expected format: {logs[3]}'
    assert re.search(r'het_foa_logs-\d+-\d+-agentreplacements', logs[4]), f'5th log does not match expected format: {logs[4]}'

    return Timestep(
        timestep=timestep,
        input_states=get_states_from_log(logs[0]),
        agent_output_states=get_states_from_log(logs[1]),
        state_wins=get_py_list(logs[2].split('statewins: ')[-1].strip(), bool),
        state_fails=get_py_list(logs[3].split('statefails: ')[-1].strip(), bool),
        replacement_states=get_states_from_log(logs[4]),
        values=None
    )
    


# process the logs
logs = load_logs()
logs = logs.split('\n')
het_foa_logs = []
fleet = []
for log in logs:
    if 'het_foa_logs' in log:
        if '-fleet:' in log:
            if len(fleet) == 0:
                fleet = get_fleet(log)
            else:
                assert fleet == get_fleet(log), f'Fleet mismatch in log: {log} and {fleet=}'
        else:
            het_foa_logs.append('het_foa_logs: ' + log.split('het_foa_logs: ')[-1].strip())

puzzles = set()
for log in het_foa_logs:
    puzzles.add(get_puzzle_idx(log))

puzzles = {
    pid: []
    for pid in list(puzzles)
}

for log in het_foa_logs:
    puzzle_idx = get_puzzle_idx(log)
    puzzles[puzzle_idx].append(log)


graph: Dict[int, List[Timestep]] = {}
flows = {}
for puzzle_idx, logs in puzzles.items():
    graph[puzzle_idx] = []
    t = 0
    while len(logs) > 0:
        if len(logs) == 5:
            timestep = get_final_timestep(logs, t)
            logs = logs[5:]
        else:
            timestep = get_timestep_object(logs[:6], t)
            logs = logs[6:]
        graph[puzzle_idx].append(timestep)
        t += 1

    num_colors = len(state_colors)
    colors = generate_distinct_hex_colors(num_colors)
    random.shuffle(colors)

    for k in state_colors:
        state_colors[k] = colors.pop(0)

    # iterate over all States and reset colors
    for timestep in graph[puzzle_idx]:
        for state in timestep.input_states + timestep.agent_output_states + timestep.replacement_states:
            state.color = get_state_color(state.name)

    for timestep in graph[puzzle_idx]:
        for i in range(len(timestep.agent_output_states)):
            if timestep.state_fails[i]:
                timestep.agent_output_states[i].terminal_data = 'Failed'
            elif timestep.state_wins[i]:
                timestep.agent_output_states[i].terminal_data = 'Winning'
            
            if timestep.agent_output_states[i].value is None:
                timestep.agent_output_states[i].value = timestep.values[i] if timestep.values else None

    flows[puzzle_idx] = [{
        'agent_name': fleet[i],
        'input_states': [t.input_states[i] for t in graph[puzzle_idx]],
        'output_states': [t.agent_output_states[i] for t in graph[puzzle_idx]],
    } for i in range(len(fleet))]



def draw_agent_diagram(agent_name: str, input_states: List[State], output_states: List[State], 
                      x_offset: int = 0, font_size: int = 14) -> tuple[Image.Image, int]:
    """
    Draw a single agent diagram and return the image and the width used.
    """
    # Configuration
    padding = 20
    state_width = 200
    state_padding = 10
    arrow_height = 30
    spacing_between_pairs = 40
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
        bold_font = ImageFont.truetype("arialbd.ttf", font_size)
    except:
        # Fallback to default font if system fonts aren't available
        font = ImageFont.load_default()
        bold_font = font
    
    # Calculate dimensions
    max_pairs = max(len(input_states), len(output_states))
    
    # Calculate height needed
    agent_name_height = 40
    state_height = 100  # Base height for state rectangles
    total_height = (padding * 2 + 
                   agent_name_height + 
                   max_pairs * (state_height * 2 + arrow_height + spacing_between_pairs))
    
    # Calculate width needed
    diagram_width = state_width + padding * 2
    
    # Create image
    img = Image.new('RGB', (diagram_width, total_height), 'white')
    draw = ImageDraw.Draw(img)
    
    current_y = padding
    
    # Draw agent name in black rectangle
    agent_rect = (x_offset + padding, current_y, 
                  x_offset + padding + state_width, current_y + agent_name_height)
    draw.rectangle(agent_rect, fill='black')
    
    # Center the agent name text
    agent_text_bbox = draw.textbbox((0, 0), agent_name, font=bold_font)
    agent_text_width = agent_text_bbox[2] - agent_text_bbox[0]
    agent_text_height = agent_text_bbox[3] - agent_text_bbox[1]
    agent_text_x = x_offset + padding + (state_width - agent_text_width) // 2
    agent_text_y = current_y + (agent_name_height - agent_text_height) // 2
    draw.text((agent_text_x, agent_text_y), agent_name, fill='white', font=bold_font)
    
    current_y += agent_name_height + padding
    
    # Draw state pairs
    for i in range(max_pairs):
        # Draw input state if exists
        if i < len(input_states):
            current_y = draw_state(draw, input_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        # Draw arrow
        arrow_start_x = x_offset + padding + state_width // 2
        arrow_start_y = current_y + 5
        arrow_end_y = current_y + arrow_height - 5
        
        # Arrow shaft
        draw.line([(arrow_start_x, arrow_start_y), (arrow_start_x, arrow_end_y)], 
                 fill='black', width=2)
        
        # Arrow head
        arrow_head_size = 5
        draw.polygon([(arrow_start_x, arrow_end_y),
                     (arrow_start_x - arrow_head_size, arrow_end_y - arrow_head_size),
                     (arrow_start_x + arrow_head_size, arrow_end_y - arrow_head_size)],
                    fill='black')
        
        current_y += arrow_height
        
        # Draw output state if exists
        if i < len(output_states):
            current_y = draw_state(draw, output_states[i], x_offset + padding, current_y, 
                                 state_width, font, bold_font, state_padding)
        
        current_y += spacing_between_pairs
    
    return img, diagram_width

def draw_state(draw: ImageDraw.Draw, state: State, x: int, y: int, width: int, 
               font: ImageFont.ImageFont, bold_font: ImageFont.ImageFont, padding: int) -> int:
    """
    Draw a single state rectangle and return the y position after drawing.
    """
    # Calculate text lines
    lines = [state.name]  # Bold line
    
    if state.value is not None:
        lines.append(f"Value: {state.value}")
    
    if state.num_thoughts > 0:
        lines.append(f"Thoughts: {state.num_thoughts}")

    if len(state.terminal_data) > 0:
        lines.append(f"{state.terminal_data} State")
    
    # Calculate height needed
    line_height = 20
    text_height = 4 * line_height
    total_height = text_height + padding * 2
    
    # Draw state rectangle
    state_rect = (x, y, x + width, y + total_height)
    draw.rectangle(state_rect, fill=state.color, outline='black', width=1)
    
    # Draw text lines
    text_y = y + padding
    for i, line in enumerate(lines):
        current_font = bold_font if i == 0 else font  # First line (name) is bold
        draw.text((x + padding, text_y), line, fill='black', font=current_font)
        text_y += line_height
    
    return y + total_height

def create_agent_diagrams(diagrams_data: List[dict], spacing: int = 50) -> Image.Image:
    """
    Create multiple agent diagrams in a single image.
    
    diagrams_data: List of dictionaries with keys 'agent_name', 'input_states', 'output_states'
    spacing: Horizontal spacing between diagrams
    """
    if not diagrams_data:
        return Image.new('RGB', (100, 100), 'white')
    
    # First pass: calculate individual diagram dimensions
    diagram_images = []
    diagram_widths = []
    max_height = 0
    
    for data in diagrams_data:
        img, width = draw_agent_diagram(
            data['agent_name'], 
            data['input_states'], 
            data['output_states']
        )
        diagram_images.append(img)
        diagram_widths.append(width)
        max_height = max(max_height, img.height)
    
    # Calculate total width
    total_width = sum(diagram_widths) + spacing * (len(diagrams_data) - 1)
    
    # Create final image
    final_image = Image.new('RGB', (total_width, max_height), 'white')
    
    # Paste diagrams
    current_x = 0
    for i, img in enumerate(diagram_images):
        final_image.paste(img, (current_x, 0))
        current_x += diagram_widths[i] + spacing
    
    return final_image


current_puzzle = None
while True:
    cmd = input('>>> ')

    if cmd == 'q':
        break

    if cmd == 'clear':
        os.system('cls' if os.name == 'nt' else 'clear')
        continue

    if cmd.startswith('open '):
        puzzle_idx = int(cmd.split(' ')[1])
        if puzzle_idx not in flows:
            print(colored(f'Puzzle {puzzle_idx} not found.', 'red'))
            continue
        
        current_puzzle = puzzle_idx
        continue

    if cmd.startswith('img'):
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue
        
        img = create_agent_diagrams(flows[current_puzzle])
        os.makedirs('tmp', exist_ok=True)
        img.save(f'tmp/pic_{current_puzzle}.png', format='PNG')
        print(colored(f'Image saved as tmp/pic_{current_puzzle}.png', 'green'))
        continue

    if cmd == 'ls':
        for puzzle_idx in flows:
            print(f'Puzzle {puzzle_idx}: ', colored('Won', 'green') if any(graph[puzzle_idx][-1].state_wins) else colored('Failed', 'red'))
        continue

    res = re.search(f'^s(\d+).*$', cmd)
    if res:
        idx = int(res.group(1))
        if current_puzzle is None:
            print(colored('No puzzle selected. Use "open <puzzle_idx>" to select a puzzle.', 'red'))
            continue

        name = f's{idx}'
        if name not in state_names.values():
            print(colored(f'State {name} not found.', 'red'))
            continue

        # Find the state in the current puzzle
        found = False
        state = None
        for timestep in reversed(graph[current_puzzle]):
            for s in timestep.agent_output_states:
                if s.name == name:
                    state = s
                    found = True
                    break

            if found:
                break
        
        
        if not found:
            for s in graph[current_puzzle][0].input_states:
                if s.name == name:
                    state = s
                    found = True
                    break

        if not found:
            print(colored(f'State {name} not found in puzzle {current_puzzle}.', 'red'))
            continue

        attr = cmd.replace(f's{idx}.', '').strip()
        attr = attr.replace('cs', 'current_state') # shorthand
        attr = attr.replace('sd', 'serial_data') # shorthand

        try:
            if any(attr.startswith(field) for field in ['name', 'color', 'num_thoughts', 'value', 'terminal_data', 'serial_data']):
                expr = f'state.{attr}'
            else:
                expr = f'state.serial_data["{attr}"]'
            
            print(eval(expr))
        except:
            print(colored(f'Attribute {attr} not found in state {name}.', 'red'))

        continue


    print(colored('Unknown command.', 'yellow'))