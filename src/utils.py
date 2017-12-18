import string

name_cleaner = lambda n: n.split()[0].translate(str.maketrans('', '', string.punctuation))