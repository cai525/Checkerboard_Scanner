import ctypes
import os
ll = ctypes.cdll.LoadLibrary
lib = ll('./ct.so')
lib.hehe()
