import os
from fnmatch import fnmatchcase
from toolz import curry, flip
import platform
import urllib.request
import zipfile
from zipfile import ZipFile

def match_filename(directory, pattern):
    contents = list(filter(curry(flip(fnmatchcase))(pattern), os.listdir(directory)))
    if len(contents) > 1:
        raise ValueError('File pattern is ambiguous.')
    if not contents:
        return None
    return contents[0]

def unzip_and_fix_permissions(zippath, directory):
    '''
    Unzip with all permissions set to default.  This is needed because 
    the permissions in Banana.app are not set correctly as archived, causing
    the environment to fail to load.  It's necessary to do it this way so the
    program doesn't need to be run with sudo.
    
    Modified from https://stackoverflow.com/a/596455/1572508.
    '''
    infile = ZipFile(zippath, 'r')
    for file in infile.filelist:
        name = file.filename
        if name.endswith('/'):
            outfile = os.path.join(directory, name)
            try:
                os.mkdir(outfile)
            except:
                os.mkdir(outfile)
        else:
            outfile = os.path.join(directory, name)
            fh = os.open(outfile, (os.O_CREAT | os.O_WRONLY))
            os.write(fh, infile.read(name))
            os.close(fh)

def download_banana(directory):
    '''
    Download the correct compiled Banana environment for this platform, extract 
    the zip archive, and fix all file permissions.
    '''
    system = platform.system()
    machine = platform.machine()
    if system == 'Linux':
        url = 'https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip'
    elif system == 'Darwin':
        url = 'https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip'
    elif system == 'Windows':
        if machine == 'AMD64' or machine == 'x86_64':
            url = 'https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip'
        else:
            url = 'https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip'
    print('Downloading banana environment...', end='')
    path = os.path.join(directory, url.split('/')[-1])
    urllib.request.urlretrieve(url, path) 
    print('Download complete.')
    print('Extracting banana environment...', end='')
    try:
        unzip_and_fix_permissions(path, directory)
    except:
        # Possibly on some systems the above will both fail and 
        # be unnecessary.  Then do it this way, which will not 
        # attempt to alter permissions.
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(directory)
    os.remove(path)
    print('Extraction complete.')

resources = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')
match = match_filename(resources, 'Banana*')
if match is None:
    download_banana(resources)
    match = match_filename(resources, 'Banana*')

banana = os.path.join(resources, match)

