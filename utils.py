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
	for function in functions:
		try:
			return function()
		except TypeError:
			pass
		except:
			raise

class Function:
	def __init__(self, function, cast = True):
		self.function = function
		self.cast = cast
	def __and__(self, other):
		return Function(lambda *args, **kwargs: self.function(*(args + (other,)), **kwargs), self.cast)
	def __rand__(self, other):
		return Function(lambda *args, **kwargs: self.function(other, *args, **kwargs), self.cast)
	def __call__(self, *args, **kwargs):
		return self.function(*(list(map(f, args)) if self.cast else args), **kwargs)

IDENT = None

def setident(ident):
	global IDENT
	IDENT = ident

def f(x):
	if isinstance(x, IDENT):
		return x()
	elif hasattr(x, '__iter__') and not isinstance(x, str):
		try:
			return type(x)(map(f, x))
		except:
			return x
	return x
