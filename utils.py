def FT(key, function = lambda x: x):
	print(function(key))
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

def call(func, *args, **kwargs):
	if isinstance(func, (list, tuple, set)):
		return type(func)(call(f, *args, **kwargs) for f in func)
	return func(*args, **kwargs)

funcskey = []
funcsval = []

def getfunction(function, cast = True, cache = False, partial = False):
	try:
		return funcsval[funcskey.index(function)]
	except:
		func = Function(function, cast, cache, partial)
		funcskey.append(function)
		funcsval.append(func)
		return func

class _splat:
	def __init__(self, array):
		self.array = array
	def __iter__(self):
		return iter(self.array)

def splat(obj):
	if hasattr(obj, '__iter__'):
		return _splat(obj)
	return obj

def determine_arguments(array):
	result = []
	for element in array:
		if isinstance(element, _splat):
			result += determine_arguments(element.array)
		else:
			result.append(element)
	return result

class Function:
	def __init__(self, function, cast = True, cache = False, partial = False):
		global id
		self.function = function
		self.cast = cast
		self.cache = cache
		self.partial = partial
		self.id = id
		id += 1
		if not haskey(self.function):
			fkey.append(self.function)
			keys.append([])
			fval.append(self.function)
			vals.append([])
	def __rshift__(self, other):
		return Function(lambda *args, **kwargs: call(other, *((call(self.function, *args, **kwargs),) + args), **kwargs))
	def __lshift__(self, other):
		return Function(lambda *args, **kwargs: call(self.function, *(args + (call(other, *args, **kwargs),)), **kwargs))
	def __and__(self, other):
		return Function(lambda *args, **kwargs: self.function(*(args + (other,)), **kwargs), self.cast, self.cache and other.cache)
	def __rand__(self, other):
		return Function(lambda *args, **kwargs: self.function(other, *args, **kwargs), self.cast, self.cache and other.cache)
	def __rdiv__(self, other):
		return list(reduce(self, other))
	def __call__(self, *args, **kwargs):
		if self.cast: args = cast(args)
		args = determine_arguments(args)
		if self.cache and (args, kwargs) in getkey(self.function):
			return getval(self.function, getkey(self.function).index((args, kwargs)))
		else:
			if isinstance(args, map): args = list(args)
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
	def __rmul__(self, times):
		return self * times
	def __mul__(self, times):
		times = f(times)
		if isinstance(times, Function):
			def inner(*args, **kwargs):
				value = self(*args, **kwargs)
				while not times(value):
					value = self(*args, **kwargs)
				return value
			return inner
		if times == 0: return lambda *args, **kwargs: args[0] if len(args) == 1 else args
		if times == 1: return self
		if times == 2: return self + self
		return self + self * (times - 1)
	def setCaching(self, cache):
		self.cache = cache
		return self
	def wipeCache(self):
		keys[fkey.index(self.function)] = []
		vals[fkey.index(self.function)] = []
		return self
	def setPartial(self, partial):
		self.partial = partial

IDENT = None
EVALP = None

def setident(ident):
	global IDENT
	IDENT = ident

def setevalp(evalp):
	global EVALP
	EVALP = evalp

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
		except:
			return g(x)
	else:
		return g(x)

def h(x):
	if not isinstance(x, str): return x
	result = ''
	index = 0
	while index < len(x):
		if x[index] == '\\':
			result += eval('"\\' + x[index + 1] + '"')
			index += 1
		elif x[index] == '#':
			if x[index + 1] == '{':
				string = ''
				index += 2
				while x[index] != '}':
					if x[index] == '\\':
						string += x[index + 1]
						index += 1
					else:
						string += x[index]
					index += 1
				result += str(EVALP(string))
			else:
				result += '#'
		else:
			result += x[index]
		index += 1
	return result

def make_type(dict):
	funcs = {}
	other = {}
	for key in dict:
		value = dict[key]
		if isinstance(value, Function):
			funcs[key] = value
		else:
			other[key] = value
	class InnerType:
		def __init__(self):
			for key in other:
				self.__setattr__(key, other[key])
	for key in funcs:
		setattr(InnerType, key, dict[key])
	return InnerType()
