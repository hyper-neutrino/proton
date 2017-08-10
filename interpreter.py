import parser, lexer, operator, math, sys, builtins

def f(x):
	if isinstance(x, Identifier):
		return x()
	elif hasattr(x, '__iter__') and not isinstance(x, str):
		return type(x)(map(f, x))
	return x

def _(op):
	return lambda x, y: op(f(x), f(y))

def __(op):
	return lambda x, y: x(op(f(x), f(y)))

def subref(x, y):
	ident = isinstance(y, Identifier)
	y = y.name if ident else y
	def getter():
		return getattr(f(x), y) if ident and hasattr(f(x), y) else f(x)[y]
	def setter(*args):
		if ident and hasattr(f(x), y):
			setattr(f(x), y, args[0])
		else:
			f(x)[y] = args[0]
	return Identifier('', getter, setter)

def assign(x, y):
	if hasattr(x, '__iter__'):
		if hasattr(y, '__iter__'):
			if len(x) < len(y):
				raise RuntimeError('Too many values to unpack (expected %d)' % len(x))
			elif len(x) > len(y):
				raise RuntimeError('Not enough values to unpack (needed %d)' % len(x))
			else:
				for i in range(len(x)):
					assign(x[i], y[i])
				return x
		else:
			raise RuntimeError('Right side of assignment must be iterable if left side is iterable')
	else:
		return x(f(y))


infix_operators = {
	'**': _(operator.pow),
	'*': _(operator.mul),
	'/': _(operator.truediv),
	'//': _(operator.floordiv),
	'+': _(operator.add),
	'-': _(operator.sub),
	'>>': _(operator.rshift),
	'<<': _(operator.lshift),
	'>': _(operator.gt),
	'<': _(operator.lt),
	'<=': _(operator.le),
	'>=': _(operator.ge),
	'==': _(operator.eq),
	'+=': __(operator.add),
	'in': __(lambda x, y: x in y),
	'=': assign,
	'.': subref
}

def ___(op):
	return lambda x: op(f(x))

prefix_operators = {
	'!': ___(operator.not_)
}

postfix_operators = {
	'!': ___(lambda x: type(x)(math.gamma(x + 1)))
}

class Identifier:
	def __init__(self, name, getter, setter):
		self.name = name
		self.getter = getter
		self.setter = setter
	def __call__(self, *args):
		if args: self.setter(*args)
		return self.getter()

def listify(func):
	return lambda *args, **kwargs: list(func(*args, **kwargs))

default = {

}

for name in dir(builtins):
	try:
		default[name] = eval(name)
	except:
		pass

def hardeval(tree, symlist = None, comma_mode = tuple):
	return f(evaluate(tree, symlist, comma_mode))

def evaluate(tree, symlist = None, comma_mode = tuple):
	symlist = symlist or default
	treetype = tree.type.split('/')
	tokentype = tree.token.type.replace(':', '/').split('/')
	if 'literal' in tokentype:
		return tree.token.content
	elif 'identifier' in tokentype:
		def getter():
			if tree.token.content in symlist:
				return symlist[tree.token.content]
			return 0
		def setter(*args):
			symlist[tree.token.content] = args[0]
		return Identifier(tree.token.content, getter, setter)
	elif tree.token.content == 'unless':
		if hardeval(tree.children[1], symlist):
			evaluate(tree.children[2], symlist)
		else:
			evaluate(tree.children[0], symlist)
	elif tree.token.content == 'if':
		if hardeval(tree.children[0], symlist):
			evaluate(tree.children[1], symlist)
		elif len(tree.children) > 2:
			evaluate(tree.children[2], symlist)
	elif 'binary_operator' in tokentype or 'binary_RTL' in tokentype:
		if tree.token.content in infix_operators:
			return infix_operators[tree.token.content](evaluate(tree.children[0], symlist), evaluate(tree.children[1], symlist))
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif 'prefix' in treetype:
		if tree.token.content in prefix_operators:
			return prefix_operators[tree.token.content](evaluate(tree.children[0], symlist))
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif 'postfix' in treetype:
		if tree.token.content in postfix_operators:
			return postfix_operators[tree.token.content](evaluate(tree.children[0], symlist))
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif 'comma_expr' in treetype:
		return comma_mode([evaluate(child, symlist) for child in tree.children])
	elif 'bracket_expr' in treetype:
		if 'bracket' in treetype:
			return evaluate(tree.children[0], symlist, tuple)
		elif 'list' in treetype:
			return evaluate(tree.children[0], symlist, list)
		elif 'codeblock' in treetype:
			for statement in tree.children:
				evaluate(statement, symlist)
	elif 'call' in treetype:
		return hardeval(tree.children[0], symlist)(*[hardeval(child, symlist) for child in tree.children[1:]])
	elif 'getitem' in treetype:
		def getter():
			return hardeval(tree.children[0], symlist)[hardeval(tree.children[1], symlist)]
		def setter(*args):
			hardeval(tree.children[0], symlist)[hardeval(tree.children[1], symlist)] = args[0]
		return Identifier('', getter, setter)
	elif 'expression' in treetype:
		return evaluate(tree.children[0], symlist)

class Interpreter:
	def __init__(self, tree):
		self.tree = tree
	def interpret(self, symlist = None):
		symlist = symlist or {x: default[x] for x in default}
		resut = None
		for tree in self.tree:
			result = hardeval(tree, symlist)
		return result

def complete(tree, symlist = None):
	return Interpreter(tree).interpret(symlist)
