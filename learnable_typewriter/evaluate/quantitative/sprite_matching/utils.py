import string

def sort_dict(dictionnary):
    return(dict(sorted(dictionnary.items(), key=lambda dictionnary: dictionnary[1])))

def sort_mapping(mapping, K):

    lower_case, upper_case, punctuation  = {}, {}, {}
    for k in range(K):
        char = mapping.get(k, '_')
        if char in string.punctuation:
            punctuation[k] = char
        else:
            if char.isupper():
                upper_case[k] = char
            else:
                lower_case[k] = char
    return {**sort_dict(lower_case), **sort_dict(upper_case), **punctuation}