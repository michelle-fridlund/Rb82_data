import os
                             
def myrename(original_name, new_name, dirpath):
    filepath = os.path.join(dirpath, original_name)
    try:
        os.rename(filepath, os.path.join(dirpath, new_name))
    except FileNotFoundError:
        print(f'File {filepath} not found. Skipping...')
                          
       
def find_dir(current_path):            
    for (dirpath, dirnames, filenames) in walk(current_path): 
        print(f'We are at {dirpath}')
        myrename('test1.txt', 'michelle.txt', dirpath)
        myrename('test2.txt', 'tolik.txt', dirpath)

    
#def find_dir(current_path):
#    print('We are here: ', current_path)
#    myrename('test1.txt', 'michelle.txt', current_path)
#    myrename('test2.txt', 'tolik.txt', current_path)
#    for element in os.listdir(current_path):
#        fullpath = os.path.join(current_path, element)
#        print(fullpath)
#        if (os.path.isdir(fullpath)):
#            print(f'{fullpath} is a directory, going deeper...')
#            find_dir(fullpath)

path ='/home/quantumcoke/test'
find_dir(path)