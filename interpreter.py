import parser, lexer, operator, math, sys, builtins

from utils import *

def _s(op):
	def inner(x, y):
		return safechain([lambda: op(x, y), lambda: op(x, f(y)), lambda: op(f(x), y), lambda: op(f(x), f(y))])
	return inner

def _(op):
	return lambda x, y, z: _s(op)(evaluate(x, z), evaluate(y, z))

def __(op):
	return lambda x, y, z: (lambda k, j: k(_s(op)(k, j)))(evaluate(x, z), evaluate(y, z))

def subref(x, y, z):
	x, y = evaluate(x, z), evaluate(y, z)
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

def inverse(function):
	return lambda *args, **kwargs: not function(*args, **kwargs)

def contains(x, y, s):
	if isinstance(x, parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'and':
		return contains(x.children[0], y, s) and contains(x.children[1], y, s)
	elif isinstance(x, parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'or':
		return contains(x.children[0], y, s) or contains(x.children[1], y, s)
	elif isinstance(y, parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'and':
		return contains(x, y.children[0], s) and contains(x, y.children[1], s)
	elif isinstance(y, parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'or':
		return contains(x, y.children[0], s) or contains(x, y.children[1], s)
	else:
		return f(evaluate(x, s)) in f(evaluate(y, s))

infix_operators = {
	'**': _(operator.pow),
	'>>': _(operator.rshift),
	'<<': _(operator.lshift),
	'*': _(operator.mul),
	'/': _(operator.truediv),
	'//': _(operator.floordiv),
	'%': _(operator.mod),
	'+': _(operator.add),
	'-': _(operator.sub),
	'>': _(operator.gt),
	'<': _(operator.lt),
	'<=': _(operator.le),
	'>=': _(operator.ge),
	'==': _(operator.eq),
	'in': contains,
	'not in': inverse(contains),
	'&': _(operator.and_),
	'|': _(operator.or_),
	'^': _(operator.xor),
	'&&'  : lambda x, y, z: (lambda k: k if not k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'and' : lambda x, y, z: (lambda k: k if not k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'||'  : lambda x, y, z: (lambda k: k   if   k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'or'  : lambda x, y, z: (lambda k: k   if   k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'**=': __(operator.pow),
	'*=': __(operator.mul),
	'/=': __(operator.truediv),
	'//=': __(operator.floordiv),
	'+=': __(operator.add),
	'-=': __(operator.sub),
	'>>=': __(operator.rshift),
	'<<=': __(operator.lshift),
	'%=': __(operator.mod),
	'&=': __(operator.and_),
	'|=': __(operator.or_),
	'&&=': __(lambda x, y: x and y),
	'||=': __(lambda x, y: x and y),
	'=': __(lambda x, y: y),
	'.': subref
}

def ___(op):
	return lambda x, z: op(f(evaluate(x, z)))

def exists(value, symlist):
	if value.token.type == 'binary_operator' and value.token.content in ['and', '&&']:
		return exists(value.children[0], symlist) and exists(value.children[1], symlist)
	elif value.token.type == 'binary_operator' and value.token.content in ['or', '||']:
		return exists(value.children[0], symlist) or exists(value.children[1], symlist)
	elif 'bracket' in value.type and 'bracket_expr' in value.type:
		return exists(value.children[0], symlist)
	else:
		ident = evaluate(value, symlist)
		if isinstance(ident, Identifier):
			return ident.name in symlist or ident.name == '.getitem' and cando(ident) or ident() is not None
		return True

prefix_operators = {
	'!': ___(operator.not_),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)),
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z))
}

postfix_operators = {
	'!': ___(lambda x: type(x)(math.gamma(x + 1))),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)) - 1,
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z)) + 1,
	'exists': exists,
	'exists not': inverse(exists)
}

class Identifier:
	def __init__(self, name, getter, setter):
		self.name = name
		self.getter = getter
		self.setter = setter
	def __call__(self, *args):
		if args: self.setter(*args)
		return self.getter()

setident(Identifier)

def listify(func):
	return lambda *args, **kwargs: list(func(*args, **kwargs))

default = {

}

for name in dir(builtins):
	try:
		default[name] = eval(name)
		if isinstance(default[name], (type(print), type(lambda:0), type(type))):
			default[name] = Function(default[name])
	except:
		pass

default['eval'] = lambda s, x: complete(parser.parse(lexer.tokenize(x)), s)

def clone(obj):
	if isinstance(obj, (tuple, list, set)):
		return type(obj)(map(clone, obj))
	elif isinstance(obj, dict):
		return {elem: clone(obj[elem]) for elem in obj}
	else:
		return obj

def merge(d1, d2):
	for x in d1:
		if x in d2:
			d2[x] = d1[x]

def hardeval(tree, symlist = None, comma_mode = tuple):
	return f(evaluate(tree, symlist, comma_mode))

def _hardeval(tree, symlist = None, comma_mode = tuple):
	return f(_evaluate(tree, symlist, comma_mode))

def _evaluate(tree, symlist = None, comma_mode = tuple):
	sidelist = clone(symlist)
	value = evaluate(tree, sidelist, comma_mode)
	merge(sidelist, symlist)
	return value

def evaluate(tree, symlist = None, comma_mode = tuple):
	if not isinstance(tree, parser.ASTNode):
		return tree
	symlist = symlist or default
	treetype = tree.type.split('/')
	tokentype = tree.token.type.replace(':', '/').split('/')
	if 'literal' in tokentype:
		return tree.token.content
	elif 'identifier' in tokentype:
		def getter():
			return None if tree.token.content not in symlist else symlist[tree.token.content]
		def setter(*args):
			symlist[tree.token.content] = args[0]
		return Identifier(tree.token.content, getter, setter)
	elif tree.token.content == 'unless':
		if hardeval(tree.children[1], symlist):
			_evaluate(tree.children[2], symlist)
		else:
			_evaluate(tree.children[0], symlist)
	elif tree.token.content == 'if':
		if hardeval(tree.children[0], symlist):
			_evaluate(tree.children[1], symlist)
		elif len(tree.children) > 2:
			_evaluate(tree.children[2], symlist)
	elif tree.token.content == 'while':
		if 'whileas' in treetype:
			sidelist = clone(symlist)
			value = hardeval(tree.children[0], sidelist)
			while value:
				assign(evaluate(tree.children[1], sidelist), value)
				evaluate(tree.children[2], sidelist)
				value = hardeval(tree.children[0], sidelist)
			del sidelist[evaluate(tree.children[1]).name]
			merge(sidelist, symlist)
		else:
			while hardeval(tree.children[0], symlist):
				_evaluate(tree.children[1], symlist)
	elif tree.token.content == 'for':
		if 'foreach' in treetype:
			sidelist = clone(symlist)
			for val in hardeval(tree.children[1], sidelist):
				assign(evaluate(tree.children[0], sidelist), val)
				evaluate(tree.children[2], sidelist)
			del sidelist[evaluate(tree.children[0]).name]
			merge(sidelist, symlist)
		else:
			sidelist = clone(symlist)
			evaluate(tree.children[0], sidelist)
			while not tree.children[1].children or hardeval(tree.children[1], sidelist):
				evaluate(tree.children[-1], sidelist)
				if len(tree.children) >= 4: evaluate(tree.children[2], sidelist)
	elif tree.token.content == 'try':
		try:
			_hardeval(tree.children[0], symlist)
		except Exception as e:
			sidelist = clone(symlist)
			assign(evaluate(tree.children[1], sidelist), e)
			evaluate(tree.children[2], sidelist)
	elif 'binary_operator' in tokentype or 'binary_RTL' in tokentype:
		if tree.token.content in infix_operators:
			return infix_operators[tree.token.content](tree.children[0], tree.children[1], symlist)
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif 'prefix' in treetype:
		if tree.token.content in prefix_operators:
			return prefix_operators[tree.token.content](tree.children[0], symlist)
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif 'postfix' in treetype:
		if tree.token.content in postfix_operators:
			return postfix_operators[tree.token.content](tree.children[0], symlist)
		else:
			raise NotImplementedError(tree.token.content + ' not yet implemented')
	elif tree.token.content == 'exist':
		return exists(tree.children[0], symlist)
	elif 'comma_expr' in treetype:
		return comma_mode([evaluate(child, symlist) for child in tree.children])
	elif 'bracket_expr' in treetype:
		if 'opfunc' in treetype:
			return Function(lambda *args, **kwargs: infix_operators[tree.children[0].token.content](*(args + (symlist,)), **kwargs), False)
		if 'bracket' in treetype:
				return evaluate(tree.children[0], symlist, tuple)
		elif 'list' in treetype:
			if tree.children:
				result = evaluate(tree.children[0], symlist, list)
				if not isinstance(result, list): result = [result]
				return result
			return []
		elif 'codeblock' in treetype:
			sidelist = clone(symlist)
			for statement in tree.children:
				evaluate(statement, sidelist)
			merge(sidelist, symlist)
	elif tree.token.type == 'ternary':
		return evaluate(tree.children[1], symlist) if hardeval(tree.children[0], symlist) else evaluate(tree.children[2], symlist)
	elif 'call' in treetype:
		if tree.children[0].token.content in ['eval']:
			return hardeval(tree.children[0], symlist)(*([symlist] + [evaluate(child, symlist) for child in tree.children[1:]]))
		else:
			return hardeval(tree.children[0], symlist)(*[evaluate(child, symlist) for child in tree.children[1:]])
	elif 'getitem' in treetype:
		def getter():
			return hardeval(tree.children[0], symlist)[hardeval(tree.children[1], symlist)]
		def setter(*args):
			hardeval(tree.children[0], symlist)[hardeval(tree.children[1], symlist)] = args[0]
		return Identifier('.getitem', getter, setter)
	elif 'expression' in treetype:
		return evaluate(tree.children[0], symlist)

class Interpreter:
	def __init__(self, tree):
		self.tree = tree
	def interpret(self, symlist = None):
		symlist = symlist or {x: default[x] for x in default}
		result = None
		for tree in self.tree:
			result = hardeval(tree, symlist)
		return result

def complete(tree, symlist = None):
	return Interpreter(tree).interpret(symlist)
