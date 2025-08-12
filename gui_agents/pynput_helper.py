"""
Pynput helper module providing pyautogui-compatible functions
"""
from pynput.keyboard import Key, Controller
from pynput.mouse import Button, Controller as MouseController
import time
import platform

keyboard = Controller()
mouse = MouseController()

def _get_key(key_name):
    """Convert string key names to pynput Key objects"""
    key_map = {
        'command': Key.cmd,
        'cmd': Key.cmd,
        'win': Key.cmd if platform.system() == 'Darwin' else Key.cmd_l,
        'windows': Key.cmd if platform.system() == 'Darwin' else Key.cmd_l,
        'ctrl': Key.ctrl,
        'control': Key.ctrl,
        'alt': Key.alt,
        'shift': Key.shift,
        'enter': Key.enter,
        'return': Key.enter,
        'escape': Key.esc,
        'esc': Key.esc,
        'backspace': Key.backspace,
        'delete': Key.delete,
        'space': Key.space,
        'tab': Key.tab,
        'up': Key.up,
        'down': Key.down,
        'left': Key.left,
        'right': Key.right,
        'home': Key.home,
        'end': Key.end,
        'pageup': Key.page_up,
        'pagedown': Key.page_down,
        'f1': Key.f1,
        'f2': Key.f2,
        'f3': Key.f3,
        'f4': Key.f4,
        'f5': Key.f5,
        'f6': Key.f6,
        'f7': Key.f7,
        'f8': Key.f8,
        'f9': Key.f9,
        'f10': Key.f10,
        'f11': Key.f11,
        'f12': Key.f12,
    }
    
    if isinstance(key_name, str):
        key_lower = key_name.lower()
        if key_lower in key_map:
            return key_map[key_lower]
        elif len(key_name) == 1:
            return key_name
    return key_name

def hotkey(*keys, interval=0.01):
    """Press multiple keys simultaneously (pyautogui.hotkey compatible)"""
    parsed_keys = [_get_key(k) for k in keys]
    
    for key in parsed_keys:
        keyboard.press(key)
        if interval > 0:
            time.sleep(interval)
    
    for key in reversed(parsed_keys):
        keyboard.release(key)
        if interval > 0:
            time.sleep(interval)

def press(key, presses=1, interval=0.0):
    """Press and release a key (pyautogui.press compatible)"""
    parsed_key = _get_key(key)
    for _ in range(presses):
        keyboard.press(parsed_key)
        keyboard.release(parsed_key)
        if interval > 0:
            time.sleep(interval)

def keyDown(key):
    """Hold down a key (pyautogui.keyDown compatible)"""
    parsed_key = _get_key(key)
    keyboard.press(parsed_key)

def keyUp(key):
    """Release a key (pyautogui.keyUp compatible)"""
    parsed_key = _get_key(key)
    keyboard.release(parsed_key)

def write(text, interval=0.0):
    """Type text (pyautogui.write compatible)"""
    for char in text:
        keyboard.type(char)
        if interval > 0:
            time.sleep(interval)

def typewrite(text, interval=0.0):
    """Type text (alias for write, pyautogui.typewrite compatible)"""
    write(text, interval)

# Mouse functions for compatibility
def click(x=None, y=None, clicks=1, interval=0.0, button='left'):
    """Click the mouse (pyautogui.click compatible)"""
    if x is not None and y is not None:
        mouse.position = (x, y)
    
    button_map = {
        'left': Button.left,
        'right': Button.right,
        'middle': Button.middle
    }
    btn = button_map.get(button, Button.left)
    
    for _ in range(clicks):
        mouse.click(btn)
        if interval > 0:
            time.sleep(interval)

def moveTo(x, y, duration=0.0):
    """Move mouse to position (pyautogui.moveTo compatible)"""
    if duration > 0:
        start_x, start_y = mouse.position
        steps = int(duration * 50)
        for i in range(steps):
            progress = (i + 1) / steps
            new_x = start_x + (x - start_x) * progress
            new_y = start_y + (y - start_y) * progress
            mouse.position = (new_x, new_y)
            time.sleep(duration / steps)
    else:
        mouse.position = (x, y)

def scroll(clicks, x=None, y=None):
    """Scroll the mouse wheel (pyautogui.scroll compatible)"""
    if x is not None and y is not None:
        mouse.position = (x, y)
    mouse.scroll(0, clicks)

def doubleClick(x=None, y=None, interval=0.0):
    """Double click (pyautogui.doubleClick compatible)"""
    click(x, y, clicks=2, interval=interval)

def tripleClick(x=None, y=None, interval=0.0):
    """Triple click (pyautogui.tripleClick compatible)"""
    click(x, y, clicks=3, interval=interval)

def rightClick(x=None, y=None):
    """Right click (pyautogui.rightClick compatible)"""
    click(x, y, button='right')