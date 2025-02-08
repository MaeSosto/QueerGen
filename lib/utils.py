import lib.constants as lib

# The `truncate` function rounds a float number to a specified number of decimal places.
def truncate(float_number, decimal_places = 2):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier if float_number != None else 0

# The function `getListFromString` takes a string containing a list of integers and returns a list of
# integers.
def getListFromString(text):
    text = lib.re.sub(r"'", "", str(text))
    text = lib.re.sub(r'\]', '', text)
    text = lib.re.sub(r'\[', '', text)
    return list(map(int, text.split(",")))

# This Python function allows the user to choose a model from a list of options provided by the
# `lib.MODEL_LIST` and returns the index of the chosen model.
# :return: The function `chooseModel()` returns the index of the chosen model from the
# `lib.MODEL_LIST`.
def chooseModel():
    chosenModel = -1
    while chosenModel < 0 or chosenModel > len(lib.MODEL_LIST)-1:
        print('๏ Select a model: ')
        for idx, x in enumerate(lib.MODEL_LIST):
            print(f"[{idx}] -  {x}")
        chosenModel = int(input())
    return chosenModel

# The `clean_response` function removes newline characters, double quotes, and backticks from a given
# response string.
def clean_response(response):
    response = lib.re.sub(r'\n', '', response)
    response = lib.re.sub(r'\"', '', response)
    response = lib.re.sub(r'`', '', response)
    return response

# # Example usage
# s1 = "The cat is on the mat"
# s2 = "on the mat"
# result = find_subsequence_indices(s1, s2)
# Output: (3, 6)
def find_subsequence_indices(s1, s2):
    # Split sentences into word lists
    words_s1 = s1.split()
    words_s2 = s2.split()
    
    len_s2 = len(words_s2)
    
    # Iterate over s1 to find where s2 starts
    for i in range(len(words_s1) - len_s2 + 1):
        if words_s1[i:i + len_s2] == words_s2:
            return i, i + len_s2  # Start and end indices (Python-style)
    
    return None  # If s2 is not found in s1

