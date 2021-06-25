# time from string (now) #//snipped//
link : 'https://stackoverflow.com/questions/45912689/convert-time-in-hhmm-format-to-date-time-date-set-to-today'
def read_time(time_str):
    now = datetime.today()
    return datetime.strptime(time_str, '%H:%M').replace(year=now.year,month=now.month,day=now.day)
   

# base 2, binary, decimal
int(number,2) ==> binary


# matrix of size n, m
n,m = 5,10
import numpy as np
m = np.empty((n,m))

# easy print list
elements = [1,2,3,4,5,6,7]
print(*elements)
>>> 1 2 3 4 5 6 7
	or
print(a,b,c,d,sep=".")

# shift coding, alphabet
import string

def shift_by_2(msg):
    # translate lowercase letters
    alphabet = string.ascii_lowercase # 'abcdefghijklmnopqrstuvwxyz'
    trans = str.maketrans(alphabet, alphabet[2:] + alphabet[:2])
    msg = msg.translate(trans)

    # translate uppercase letters
    alphabet = string.ascii_uppercase # 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    trans = str.maketrans(alphabet, alphabet[2:] + alphabet[:2])
    msg = msg.translate(trans)

    return msg

print(shift_by_2('HeLlO tHere :)'))


#is alpha, short if else
s = 'abcfkadjc  a 54da67 ka mf= à)"'
s = input()
done=set()
o=''
for ch in s:
	if ch.isalpha() and ch not in done:
		done.add(ch)
		if s.count(ch) > 1:
			o += ch
if len(o): print(o)
else: print("NONE")

# alpha-numerical #//snipped//
there is a difference between alpha and alpha-numerical !!!!!!!
c.isalpha()   ex: "User"
c.isalnum() ex: "User123545"

#round up, down and normal round for a float #//snipped//
import math
x = 1.6
round_up = math.ceil(x)
round_down = math.floor(x)
round_ = round(x)
round_digits = round(x, number_of_digits)

#bin count simple while
n = int(input())
x=n+1
while bin(n).count('1')!=bin(x).count('1'):x+=1
print(x)


#ip to decimal
ipv_4 = '192.168.1.0'.split('.')
r = ''
for i in ipv_4:
    r += "{0:08b}".format(int(i))
print(int(r,2))


# set from other vectors
alph = set('abcdefghijklmnopqrstuvwxyz')
s = input()
for c in s.lower():
    if c in alph:
        alph.remove(c)
if len(alph)==0:
    print("true")
else:
    print("false")
	
	
# sell or buy (return max balance)
b = 100.0 #balance
t = 10 #prices (ticks)
d=[]
p=[
10.0,
20.0,
40.0,
5.0,
15.0,
10.0,
20.0,
40.0,
5.0,
11.0
]
for i in range(t-1):
    d+=[b/p[i]*max(p[i+1:])]
print(round(max(max(d),b)))

#leap year #//snipped//
y = 2020
if y%4==0 and y%100!=0 or y%400==0:print('true')
else:print('false')

#sets operations #//snipped//
x in s
x not in s

s.issubset(t)
s <= t

s.issuperset(t)
s >= t

s.union(t)
s | t

s.intersection(t)
s & t

s.difference(t)
s - t

s ^ t
s.symmetric_difference(t)
# eater s or t but not both

s.copy()

#sum of digits
n = input()
while len(n)>1:
 n=str(sum([int(d)for d in n]))
print(n)

# compare two consecutive chars without headache #previous #next
inp = '1112222223333334444455578888'
res = ''
for c in inp:
    try:
        if c != res[-1]:
            res += ' '
    except:
        pass
    res += c
print(res)

# split string to chunks of same size #//snipped//
s = '145978664258'
size = 3
chunks = [s[i:i + size] for i in range(0, n, size)]
print(*[chr(int(x)) for x in chunks],sep='')

#first vowels in words
for i in['NONE']*int(input()):
 w=input()
 for c in 'UuOoIiEeAa': i = [i,c][c in w]
 print(i)

# decimal to base b #//snipped//
def decimal_to_base(n, base):
    ''' returns a string representing the number in the given base'''
    order_of_A = ord('A')
    s=''
    while n>0:
        r=n%base
        if r<10:
            s=str(r)+s
        else:
            s+=chr(r-10+order_of_A)
        n//=base
    return s

# list empty #//snipped//
if not some_list:
    print('some_list is empty')

# quick input # fast input !!!
f=lambda:int(input())
s=f()
n=f()
p=[f()for _ in'0'*n]
print(sum(p)-max(p)*s//100)

#rgb #colors #hex #hex to decimal dec
color = "#FFA67B"
r,g,b=color[1:3],color[3:5],color[5:7]
print(*[int("0x"+r,base=16),int("0x"+g,base=16),int("0x"+b,base=16)],sep='\n')


# rotation, slicing, array, rotate
def rotate(array, start):
    ''' [A B C D E F G H] , start from 2 ('C') --> [C D E F G H] + [A B] = [C D E F G H A B] '''
    return array[start:] + array[:start]

# replace many chars at once, translate
import string
coded_msg = "#@²²*"
T = str.maketrans('#@²*', 'helo')
decoded_msg = coded_msg.translate(T)




# folders and files functions   
    // moving files 
        import shutil
        import os
        
        def move_files(source, dest, remove_src = False, rename_as = ''):
            files = os.listdir(source)
            for f in files:
                shutil.move(os.path.join(source,f), dest)
            if remove_src:
                os.removedirs(source)
            if len(rename_as) > 1:
                rename_as = os.path.join(os.path.dirname(dest), rename_as)
                os.rename(dest, rename_as)
    // recursive delete / delete all
        import shutil
        shutil.rmtree('/directory/to/delete')
    // get list of all sub files !
        import ***

        def all_sub_files(path):
            inner_files = listdir(path)
            onlyfiles = [join(path, f) for f in inner_files if isfile(join(path, f))]
            for directory in [join(path,f) for f in listdir(path) if isdir(join(path, f))]:
                onlyfiles.extend(all_sub_files(directory))
            return onlyfiles


    import zipfile 
    import shutil
    import os
    import traceback

    # same, but copy2 instead
    def copy_files(source, dest, remove_src = False, rename_as = ''):
        files = os.listdir(source)
        for f in files:
            shutil.copy2(os.path.join(source,f), dest)
        if remove_src:
            os.removedirs(source)
        if len(rename_as) > 1:
            rename_as = os.path.join(os.path.dirname(dest), rename_as)
            os.rename(dest, rename_as)


    def extract_zipfile(src, dest):
        zip_ref = zipfile.ZipFile(src, 'r')
        zip_ref.extractall(dest)
        zip_ref.close()


    # make zipfile
    shutil.make_archive('/content/drive/My Drive/Colab Notebooks/Data/ImageNet_faces', 'zip', '/content/drive/My Drive/Colab Notebooks/Data/ImageNet_faces')

# random, randomness
    # random seed
    import random
    from datetime import datetime
    random.seed(datetime.now())

    # rand int between a, b inclusive (a <= randint(a,b) <=b)
    randint(a,b)

    # rand emelment from list
    random.choice(collection_or_list) ### find out !


# subgrids, numpy, slices, subarray, flatten matrix
    '''
███╗   ██╗ ██████╗ ████████╗███████╗    ██╗       
████╗  ██║██╔═══██╗╚══██╔══╝██╔════╝    ██║    ██╗
██╔██╗ ██║██║   ██║   ██║   █████╗      ██║    ╚═╝
██║╚██╗██║██║   ██║   ██║   ██╔══╝      ╚═╝    ██╗
██║ ╚████║╚██████╔╝   ██║   ███████╗    ██╗    ╚═╝
╚═╝  ╚═══╝ ╚═════╝    ╚═╝   ╚══════╝    ╚═╝      (ascii art : http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=NOTE%20!%20%3A)
     NOTE :
     always use numpy arrays to be able to prefom slices on multi dementional arrays !!!                                       
    '''
import numpy as np
array = np.array(array)

row_i = array[i] = array[i,:]
column_j = array[:,j]

square_sub_array = array[0:10, 20:30]

# flatten array, matrix to array, 2D,3D,4D .. to 1D
from itertools import chain
flat_arr = list(chain.from_iterable(array))
    

# or notation instead of if # easier if then else notation #//snipped//
THIS CODE :

x = (cond and a) or b
---------------------------------

IS EQUIVALENT TO THIS CODE !!

if cond:
    x = a
else:
    x = b
print(x)

pros: short
cons: not very readable / friendly
---------------------------------

ALSO THIS

x = a if cond else b


# generate random pasword

import random
import string
from datetime import datetime
random.seed(datetime.now())

categories = [
	string.ascii_lowercase,
	string.ascii_uppercase,
	'+-*/@#$µ&',
	'0123456789'
]

password_length = 30
password = ''
for i in range(password_length):
	cat = random.choice(categories)
	password += random.choice(cat)

print(password)

#==================================================================================================
#=== STUFF TO KNOW ==================================================================================================
#==================================================================================================
. functions ARE TREATED LIKE OBJECTS !
    . this means you can add attributes to them :
        def function():
            pass
        function.x = 10
    . also means, you can methods to them !!! :
        def function():
            def method()
                pass
    . basically, their just a small inner python code :3

#==================================================================================================
#=== Decorators ==================================================================================================
#==================================================================================================

#=== Chaining decorators ==================================================================================================
def decor_A(func): 
    def wrapper(*args, **kwargs): 
        x = func(*args, **kwargs)
        print("called decor_A")
        return x 
    return wrapper 
  
def decor_B(func): 
    def wrapper(*args, **kwargs): 
        x = func(*args, **kwargs) 
        print("called decor_B")
        return x 
    return wrapper 
  
@decor_A
@decor_B
def num(x): 
    return 100*x*x
  
print(num(3)) 


#=== Can have stuff inside your decorator ;3 ==================================================================================================
def decorator(func):

    def employer():
        print("Say something about you.")

    def say_name():
        print("My name is Guido van Rossum.")

    def say_nationality():
        print("I am from Netherlands.")

    def wrapper():
        employer()
        say_name()
        say_nationality()
        func()

    return wrapper

#=== Decorator with params, decorator factory ! ==================================================================================================
def decorator_factory(argument):
    def decorator(function):
        def wrapper(*args, **kwargs):
            funny_stuff()
            something_with_argument(argument)
            result = function(*args, **kwargs)
            more_funny_stuff()
            return result
        return wrapper
    return decorator

@decorator_factory(arg)
def function():
    pass

#==================================================================================================
#=== Useful decorators ==================================================================================================
#==================================================================================================

#--- memorize #//snipped//--------------------------------------------------------------------------------------------------

def memoize(f):
    ''' memorizes already calculated values, so makes the executon faster (useful for recursive functions like fibonacci) 
    only use on "injective" functions'''
    memo = {}
    def helper(x):
        if x not in memo:            
            tmp = memo[x] = f(x)
        return tmp
    return helper

#--- memorize with arbitrary args #//snipped//--------------------------------------------------------------------------------------------------

def memoize(f):
    ''' memorizes already calculated values, so makes the executon faster (useful for recursive functions like fibonacci) 
    only use on "injective" functions'''
    memo = {}
    def helper(*args, **kwargs):
        # transform args and kwargs into a hash-code
        x = ''
        for arg in args + kwargs.keys():
            x += str(arg)
        if x not in memo:            
            tmp = memo[x] = f(*args, **kwargs)
        return tmp
    return helper

#=== Timer ==================================================================================================
import time

def timer(func): 
    def wrapper(*args, **kwargs):
        before = time.time()
        x = func(*args, **kwargs)
        print("%s function took:" % func.__name__, time.time() - before, "seconds")
        return x 
    return wrapper

#=== log to file ==================================================================================================
import datetime

def log(func): 
    def wrapper(*args, **kwargs):
        with open('logs.txt', 'a') as f:
            f.write("Called #{} with ({}) at {}\n".format(
                func.__name__,
                ", ".join([str(x) for x in args] + ['%s=%s'%(k,str(v)) for k,v in kwargs.items()]),
                datetime.datetime.now()
                )
            )
        x = func(*args, **kwargs)
        return x 
    return wrapper

#=== run as thread :D ==================================================================================================
def run_as_thread(func):
    ''' returns the thread to the function ! so you can gather them out in a list or something :3'''
    def wrapper(*args, **kwargs):
        t = threading.Thread(target=func, args=args, kwargs=kwargs)
        t.start()
        return t
    return wrapper

#=== Synchronized :D ==================================================================================================
import threading

def synchronized(use_Rlock = False):
    ''' makes all calls to a function or a method in mutual exlusion
    works even accross many instances of a class ! thread safe if used with a method ;)'''
    def decorator(func):
        lock_name = 'sync_LOCK'
        if not hasattr(func, lock_name): 
            func.__setattr__(lock_name, threading.RLock() if use_Rlock else threading.Lock())
        def wrapper(*args, **kwargs):
            with func.sync_LOCK:
                x = func(*args, **kwargs)
            return x 
        return wrapper
    return decorator

# used like this:

@synchronized() # with or without param
def function():
    pass

or 

class SomeClass():
    def __init__(self):
        pass

    @synchronized()
    def function():
        pass

#==================================================================================================
#=== Other tricks ==================================================================================================
#==================================================================================================

#=== see if object has attribute : hasattr pre-defined function ==================================================================================================
if hasattr(obj, 'property'):
    print(True)

#=== extend dict, getattr, setattr and more #//snipped// ==================================================================================================
class struct(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

be careful ! theres also a one called __getattribute__ which works mostly, unlike __getattr__

#=== getters and setters !!!!!!! #//snipped// ==================================================================================================
link = 'https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters'

class C(object):
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        print("getter of x called")
        return self._x

    @x.setter
    def x(self, value):
        print("setter of x called")
        self._x = value

    @x.deleter
    def x(self):
        print("deleter of x called")
        del self._x


c = C()
c.x = 'foo'  # setter called
foo = c.x    # getter called
del c.x      # deleter called


#==================================================================================================
#=== folders and files functions ==================================================================================================
#==================================================================================================
# moving files 
import shutil
import os

def move_files(source, dest, remove_src = False, rename_as = ''):
    files = os.listdir(source)
    for f in files:
        shutil.move(os.path.join(source,f), dest)
    if remove_src:
        os.removedirs(source)
    if len(rename_as) > 1:
        rename_as = os.path.join(os.path.dirname(dest), rename_as)
        os.rename(dest, rename_as)

# recursive delete / delete all #//snipped//
import shutil
shutil.rmtree('/directory/to/delete')

# get list of all sub files !
import os # + sruff

def all_sub_files(path):
    inner_files = listdir(path)
    onlyfiles = [join(path, f) for f in inner_files if isfile(join(path, f))]
    for directory in [join(path,f) for f in listdir(path) if isdir(join(path, f))]:
        onlyfiles.extend(all_sub_files(directory))
    return onlyfiles



# same, but copy2 instead
import zipfile 
import shutil
import os
import traceback

def copy_files(source, dest, remove_src = False, rename_as = ''):
    files = os.listdir(source)
    for f in files:
        shutil.copy2(os.path.join(source,f), dest)
    if remove_src:
        os.removedirs(source)
    if len(rename_as) > 1:
        rename_as = os.path.join(os.path.dirname(dest), rename_as)
        os.rename(dest, rename_as)

# extract zip file # extract archive
def extract_zipfile(src, dest):
    zip_ref = zipfile.ZipFile(src, 'r')
    zip_ref.extractall(dest)
    zip_ref.close()


# make zipfile
shutil.make_archive('/content/drive/My Drive/Colab Notebooks/Data/ImageNet_faces', 'zip', '/content/drive/My Drive/Colab Notebooks/Data/ImageNet_faces')

# random, randomness
# random seed
import random
from datetime import datetime
random.seed(datetime.now())

# rand int between a, b inclusive (a <= randint(a,b) <=b)
randint(a,b)

# rand emelment from list
random.choice(collection_or_list) ### find out !

# shuffle a list in place
random.shuffle(your_list)


#==================================================================================================
#=== polynomials using decorators ==================================================================================================
#==================================================================================================

def polynomial_creator(*coeffs):
    """ coefficients are in the form a_n, a_n_1, ... a_1, a_0 
    """
    def polynomial(x):
        res = coeffs[0]
        for i in range(1, len(coeffs)):
            res = res * x + coeffs[i]
        return res
                 
    return polynomial

p1 = polynomial_creator(4)
p2 = polynomial_creator(2, 4)
p3 = polynomial_creator(1, 8, -1, 3, 2)
p4 = polynomial_creator(-1, 2, 1)


for x in range(-2, 2, 1):
    print(x, p1(x), p2(x), p3(x), p4(x))


#==================================================================================================
#=== implement your own 'with' statement ! ==================================================================================================
#==================================================================================================


link = 'https://preshing.com/20110920/the-python-with-statement-by-example/'

cr.translate(68, 68)
for i in xrange(6):
    cr.save() # entrance statemetnt
    cr.rotate(2 * math.pi * i / 6)
    cr.rectangle(-25, -60, 50, 40)
    cr.stroke()
    cr.restore() # exit statemetnt


class Saved():
    def __init__(self, cr):
        self.cr = cr
    def __enter__(self):
        self.cr.save()
        return self.cr
    def __exit__(self, type, value, traceback):
        self.cr.restore()


cr.translate(68, 68)
for i in xrange(6):
    with Saved(cr): # or saved = Saved(cr); with saved: ...... etc
        cr.rotate(2 * math.pi * i / 6)
        cr.rectangle(-25, -60, 50, 40)
        cr.stroke()

#==================================================================================================
#=== Monitor in python (prototype) ==================================================================================================
#==================================================================================================

import threading

class Monitor(object):
    def __init__(self, free_places):
        self.lock = threading.Lock() # monitor's lock (mandatory)
        #-----------------------------------------------------------------------
        '''
        your code here #
        example:
        self.cond = self.Condition()
        self.io_point_1 = self.IOpoint(parent)'''

    class IOpoint(object):
        def __init__(self, parent):
            self.lock = threading.Condition(parent.lock)

        def __enter__(self):
            with self.lock:
                # your code here #
                return
            
        def __exit__(self, type, value, traceback):
            with self.lock:
                # your code here #
                return

    def Condition(self):
        return threading.Condition(self.lock)

#usage:
    monitor = Monitor()
    with monitor.io_point_1:
        # do stuff ...
        pass
#EQUIVALENT TO:
    monitor = Monitor()
    monitor.io_point_1.__enter__()
    try:
        pass# do stuff ...
    except:
        pass# ...
    finally:
        monitor.io_point_1.__exit__()

''' 
tips:
    . notify processes when they the condition they're waiting for is true 
        if cond not true:
            call.wait()

        ...
        <cond true now>
        call.notify()

'''


#==================================================================================================
# dist geo # EXCEL # XEL # EXEL # xls 
#==================================================================================================

from math import radians, cos, sin, asin, sqrt
import re
import xlsxwriter




def dms2dd(degrees, minutes, seconds, direction=''):
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60);
    if direction == 'E' or direction == 'N':
        dd *= -1
    return dd;

def dd2dms(deg):
    d = int(deg)
    md = abs(deg - d) * 60
    m = int(md)
    sd = (md - m) * 60
    return [d, m, sd]

def parse_dms(dms):
    parts = dms.split("_")#re.split(r'[°\'" ]+', dms)
    return dms2dd(*parts[:4])



def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

class Point():
    def __init__(self, lat, lon, name='', type_ = ''):
        super(Point, self).__init__()
        self.name = name
        self.lon = parse_dms(lon)
        self.lat = parse_dms(lat)
        self.type_ = type_

    def dist(self, other):
        return haversine(self.lon, self.lat, other.lon, other.lat)

    def __str__(self):
        return str(self.name) +' : '+ str(int(self.lat)) + ' ' +str(int(self.lon))


def read_points(f_name, type_=''):
    pts = []
    with open(f_name, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            r = line.split(',')
            name = r[0].strip()
            lat  = r[1].strip()
            lon  = r[2].strip()
            try:
                pts.append(Point(lat, lon, name=name, type_ = type_))
            except Exception as err:
                print(line, f_name)
                raise err
    return pts




















daira    = read_points('Daira.txt', type_='daira')
depot    = read_points('Dépôts.txt', type_='depot')
entrepot = read_points('Entrepôts.txt', type_='entrepot')
raffin   = read_points('Raffinerie.txt', type_='raffin')

workbook = xlsxwriter.Workbook('distances.xlsx')

# formats ------------------------------------------
daira_form = workbook.add_format()
depot_form = workbook.add_format()
entrepot_form = workbook.add_format()
raffin_form = workbook.add_format()
cell_form = workbook.add_format()

formats = [daira_form, depot_form, entrepot_form, raffin_form, cell_form]
for f in formats: # all has font size = 14
    f.set_font_size(14)
    f.set_align('center')
    f.set_border(1)
for f in formats[:-1]: # columns and lines are bold
    f.set_bold()

daira_form.set_bg_color('orange')
depot_form.set_bg_color('red')
entrepot_form.set_bg_color('blue')
raffin_form.set_bg_color('green')

form_of = {
    'daira' : daira_form,
    'depot' : depot_form,
    'entrepot' : entrepot_form,
    'raffin' : raffin_form,
}
# ----------------------------------------------------


couples = [
    (daira, entrepot + depot),
    (depot, entrepot),
    (raffin, entrepot)
]

names = ['Distances Livraison', 'Distances Ravitaillement', 'Distances Approvisionnement']


for n, name in enumerate(names): # for each sheet
    sheet = workbook.add_worksheet(name) # make the sheet
    a,b = couples[n]
    for i, pos in enumerate(a): # write lines
        sheet.write(i+1, 0, pos.name, form_of[pos.type_])
    for j, pos in enumerate(b): # write columns
        sheet.write(0, j+1, pos.name, form_of[pos.type_])
    for i, a_pos in enumerate(a): # write cells
        for j, b_pos in enumerate(b):
            sheet.write(i+1,j+1,a_pos.dist(b_pos), cell_form)


workbook.close()

#==================================================================================================
#=== tkinter file dialog #file #folder #directory #pick ==================================================================================================
#==================================================================================================
from tkinter import filedialog
file = filedialog.askopenfilename(initialdir = '/')
directory = filedialog.askdirectory(initialdir = '/')
# check # if path is not None and len(path) > 0:

#==================================================================================================
#=== tkinter messagebox #prompt #box ==================================================================================================
#==================================================================================================
from tkinter import messagebox

def show_error_msg(msg, title='Erreure'):
    tk.messagebox.showerror(title=title, message=msg)

def ask_yes_or_no(msg, title='Confirmer'):
    return tk.messagebox.askyesno(title=title, message=msg)

def show_info(msg, title='Success'):
    tk.messagebox.showinfo(title=title, message=msg)

def show_warning(msg, title='Avertissement'):
    tk.messagebox.showwarning(title=title, message=msg)


#==================================================================================================
#=== play/ring alert sound on cmd #noise #alarm #console #bell ==================================================================================================
#==================================================================================================
print('\a') # ring bell on terminal


#==================================================================================================
#=== System argv #input cmd args ==================================================================================================
#==================================================================================================
import sys
len(sys.argv) # alwyas >= 1
# Argument List: ['PROGRAM_NAME.py', 'arg1', 'arg2', 'arg3', ...]

#==================================================================================================
#=== #regex #re #regular #expression ==================================================================================================
#==================================================================================================

#--- ignore case --------------------------------------------------------------------------------------------------
res = re.match(r'pattern here', text_to_match, flags=re.IGNORECASE)

#--- split using regex --------------------------------------------------------------------------------------------------
re.split(r'\s*[+*-/]\s*','a + b - c / 10', flags=re.IGNORECASE)


#==================================================================================================
#=== #assert #syntax ==================================================================================================
#==================================================================================================
assert condition, "massage if error"


#==================================================================================================
#=== #typing #declare #type ==================================================================================================
#==================================================================================================
from typing import List, Dict
def func_name(param1 : List[int, int], param2: Dict[str, int]) -> str:
    return "nothing"

#--- #test type #get class of an object #find class #typeof --------------------------------------------------------------------------------------------------
if type(x) is str:
    print("do something")


#==================================================================================================
#=== #magical #methods ==================================================================================================
#==================================================================================================
#--- Binary Operators --------------------------------------------------------------------------------------------------
Operator    Method
+           object.__add__(self, other)
-           object.__sub__(self, other)
*           object.__mul__(self, other)
//          object.__floordiv__(self, other)
/           object.__truediv__(self, other)
%           object.__mod__(self, other)
**          object.__pow__(self, other[, modulo])
<<          object.__lshift__(self, other)
>>          object.__rshift__(self, other)
&           object.__and__(self, other)
^           object.__xor__(self, other)
|           object.__or__(self, other)

#--- Extended Assignments --------------------------------------------------------------------------------------------------
Operator    Method
+=          object.__iadd__(self, other)
-=          object.__isub__(self, other)
*=          object.__imul__(self, other)
/=          object.__idiv__(self, other)
//=         object.__ifloordiv__(self, other)
%=          object.__imod__(self, other)
**=         object.__ipow__(self, other[, modulo])
<<=         object.__ilshift__(self, other)
>>=         object.__irshift__(self, other)
&=          object.__iand__(self, other)
^=          object.__ixor__(self, other)
|=          object.__ior__(self, other)

#--- Unary Operators --------------------------------------------------------------------------------------------------
Operator    Method
-           object.__neg__(self)
+           object.__pos__(self)
abs()       object.__abs__(self)
~           object.__invert__(self)
complex()   object.__complex__(self)
int()       object.__int__(self)
long()      object.__long__(self)
float()     object.__float__(self)
oct()       object.__oct__(self)
hex()       object.__hex__(self

#--- Comparison Operators --------------------------------------------------------------------------------------------------
Operator    Method
<           object.__lt__(self, other)
<=          object.__le__(self, other)
==          object.__eq__(self, other)
!=          object.__ne__(self, other)
>=          object.__ge__(self, other)
>           object.__gt__(self, other)

#--- list and objects operators --------------------------------------------------------------------------------------------------
Operator        Method
array[i] = x    __setitem__
array[i]        __getitem__
obj.attr = x    __setattr__
obj.attr        __getattr__
# be careful ! theres also a one called __getattribute__ which works mostly, unlike __getattr__

if please_ignore_this_it_simply_fixes_colors_belew_on_subim : pass
# EXAMPLE: ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 
class Building(object):
    def __init__(self, floors):
        self._floors = [None]*floors
    def __setitem__(self, floor_number, data):
        self._floors[floor_number] = data
    def __getitem__(self, floor_number):
        return self._floors[floor_number]

building1 = Building(4) # Construct a building with 4 floors
building1[0] = 'Reception'
building1[1] = 'ABC Corp'
building1[2] = 'DEF Inc'
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ 

#==================================================================================================
#=== #class #method #static ==================================================================================================
#==================================================================================================
class Calculator:
    # create addNumbers static method
    @staticmethod
    def addNumbers(x, y):
        return x + y