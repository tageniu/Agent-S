import pyautogui
from time import sleep
from pynput.keyboard import Key, Controller
sleep_time = 0.5

### === Test hotkey functionality with pyautogui === ###
for i in range(2):
    print("Round:", i)
    pyautogui.hotkey('command', 'shift', 'p')
    sleep(sleep_time)
    pyautogui.hotkey('escape')
    sleep(sleep_time)
    pyautogui.hotkey('command', ',')
    sleep(sleep_time)
    pyautogui.hotkey('command', 'w')
    sleep(sleep_time)


### === Alternative using pynput === ###
keyboard = Controller()
def press_hotkey(*keys):
    for key in keys:
        keyboard.press(key)
    for key in reversed(keys):
        keyboard.release(key)

for i in range(2):
    print("Round:", i)
    press_hotkey(Key.cmd, Key.shift, 'p')
    sleep(sleep_time)
    press_hotkey(Key.esc)
    sleep(sleep_time)
    press_hotkey(Key.cmd, ',')
    sleep(sleep_time)
    press_hotkey(Key.cmd, 'w')
    sleep(sleep_time)