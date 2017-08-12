def FT(key):
	print (key)
	return key

def cando(function):
	try:
		function()
		return True
	except:
		return False

def safechain(functions):
	error = None
	for function in functions:
		try:
			return function()
		except TypeError as e:
			error = e
		except:
			raise
	raise error

def cast(args):
	if isinstance(args, (list, tuple, set)):
		try:
			return type(args)(map(cast, args))
		except:
			try:
				return map(cast, args)
			except:
				return f(args)
	return f(args)

id = 0

fkey = []
keys = []
fval = []
vals = []

def getval(function, index):
	return vals[fval.index(function)][index]

def getvalarray(function):
	return vals[fval.index(function)]

def getkey(function):
	return keys[fkey.index(function)]

def haskey(function):
	return any(key == function for key in fkey)

class Function:
	def __init__(self, function, cast = True, cache = False):
		global id
		self.function = function
		self.cast = cast
		self.cache = False
		self.id = id
		id += 1
		if not haskey(self.function):
			fkey.append(self.function)
			keys.append([])
			fval.append(self.function)
			vals.append([])
	def __and__(self, other):
		return Function(lambda *args, **kwargs: self.function(*(args + (other,)), **kwargs), self.cast, self.cache and other.cache)
	def __rand__(self, other):
		return Function(lambda *args, **kwargs: self.function(other, *args, **kwargs), self.cast, self.cache and other.cache)
	def __call__(self, *args, **kwargs):
		if self.cast: args = cast(args)
		if self.cache and (args, kwargs) in getkey(self.function):
			return getval(self.function, getkey(self.function).index((args, kwargs)))
		else:
			result = self.function(*args, **kwargs)
			if self.cache:
				getkey(self.function).append((args, kwargs))
				getvalarray(self.function).append(result)
			return result
	def __add__(self, other):
		other = f(other)
		def inner(*args, **kwargs):
			return other(self(*args, **kwargs))
		return Function(inner)
	def setCaching(self, cache):
		self.cache = cache
		return self
	def wipeCache(self):
		keys[self.id] = []
		vals[self.id] = []
		return self

IDENT = None

def setident(ident):
	global IDENT
	IDENT = ident

def g(x):
	if isinstance(x, (type(type), type(print), type(lambda: 0))):
		return Function(x)
	else:
		return x

def f(x):
	if isinstance(x, IDENT):
		return g(x())
	elif hasattr(x, '__iter__') and not isinstance(x, str):
		try:
			return type(x)(map(f, x))
			return x
		except:
			return g(x)
	else:
		return g(x)
