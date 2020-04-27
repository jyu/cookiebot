import keyboard

def main():
    while True:
        if (keyboard.is_pressed('up')):
            print("up")
        elif (keyboard.is_pressed('h')):
            print("h")
        elif (keyboard.is_pressed('space')):
            print("space")

    

if __name__ == "__main__":
    main()