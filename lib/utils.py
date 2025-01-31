import lib.constants as lib

def truncate(float_number, decimal_places = 2):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier if float_number != None else 0

def getListFromString(text):
    text = lib.re.sub(r"'", "", str(text))
    text = lib.re.sub(r'\]', '', text)
    text = lib.re.sub(r'\[', '', text)
    return list(map(int, text.split(",")))

def chooseModel():
    chosenModel = -1
    while chosenModel < 0 or chosenModel > len(lib.MODEL_LIST)-1:
        print('๏ Select a model: ')
        for idx, x in enumerate(lib.MODEL_LIST):
            print(f"[{idx}] -  {x}")
        chosenModel = int(input())
    return chosenModel