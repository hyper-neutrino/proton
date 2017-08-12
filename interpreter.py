import parser, lexer, operator, math, sys, builtins

from utils import *
from copy import *

def _import(value, symlist):
	treetype = value.type.split('/')
	tokentype = value.token.type.replace(':', '/').split('/')
	if 'comma_expr' in treetype:
		for v in value.children: _import(v, symlist)
	elif 'expression' in treetype and 'keyword' in tokentype and value.token.content == 'as':
		if 'expression' in value.children[0].type.split('/') and 'keyword' in value.children[0].token.type.replace(':', '/').split('/') and value.children[0].token.content == 'from':
			assign = evaluate(value.children[1], symlist)
			name = evaluate(value.children[0].children[0], symlist)
			module = evaluate(value.children[0].children[1], symlist)
			assert isinstance(assign, Identifier) and isinstance(name, Identifier) and isinstance(module, Identifier)
			symlist[assign.name] = getattr(__import__(module.name), name.name)
		else:
			name, assign = evaluate(value.children[0], symlist), evaluate(value.children[1], symlist)
			assert isinstance(name, Identifier) and isinstance(assign, Identifier)
			symlist[assign.name] = __import__(name.name)
	elif 'expression' in treetype and 'keyword' in tokentype and value.token.content == 'from':
		name = evaluate(value.children[0], symlist)
		module = evaluate(value.children[1], symlist)
		assert isinstance(name, Identifier) and isinstance(module, Identifier)
		symlist[name.name] = getattr(__import__(module.name), name.name)
	elif 'identifier' in tokentype:
		value = evaluate(value, symlist)
		symlist[value.name] = __import__(value.name)
	else:
		raise SyntaxError('import statement not recognized')

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

def EQ(x, y):
	return f(x) == f(y)

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
	'==': _(EQ),
	'!=': _(inverse(EQ)),
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
	'~': ___(operator.invert),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)),
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z))
}

postfix_operators = {
	'!': ___(lambda x: type(x)(math.gamma(x + 1))),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)) - 1,
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z)) + 1
}

class Statement:
	def __init__(self, name):
		self.name = name

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

def merge(d1, d2):
	for x in d1:
		if x in d2:
			d2[x] = d1[x]

def hardeval(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	return f(evaluate(tree, symlist, comma_mode, looped, func))

def _hardeval(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	return f(_evaluate(tree, symlist, comma_mode, looped, func))

def _evaluate(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	sidelist = deepcopy(symlist)
	value = evaluate(tree, sidelist, comma_mode, looped, func)
	merge(sidelist, symlist)
	return value

def evaluate(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
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
		if hardeval(tree.children[1], symlist, looped = looped, func = func):
			_evaluate(tree.children[2], symlist, looped = looped, func = func)
		else:
			_evaluate(tree.children[0], symlist, looped = looped, func = func)
	elif tree.token.content == 'if':
		if hardeval(tree.children[0], symlist, looped = looped, func = func):
			return _evaluate(tree.children[1], symlist, looped = looped, func = func)
		elif len(tree.children) > 2:
			return _evaluate(tree.children[2], symlist, looped = looped, func = func)
	elif tree.token.content == 'while':
		if 'whileas' in treetype:
			sidelist = deepcopy(symlist)
			value = hardeval(tree.children[0], sidelist, looped = looped, func = func)
			while value:
				assign(evaluate(tree.children[1], sidelist), value, looped = looped, func = func)
				result = evaluate(tree.children[2], sidelist, looped = looped, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
				value = hardeval(tree.children[0], sidelist, looped = looped, func = func)
			del sidelist[evaluate(tree.children[1], looped = looped, func = func).name]
			merge(sidelist, symlist)
		else:
			while hardeval(tree.children[0], symlist, looped = looped, func = func):
				result = _evaluate(tree.children[1], symlist, looped = looped, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
	elif tree.token.content == 'for':
		if 'foreach' in treetype:
			sidelist = deepcopy(symlist)
			for val in hardeval(tree.children[1], sidelist, looped = looped, func = func):
				assign(evaluate(tree.children[0], sidelist, looped = looped, func = func), val)
				result = evaluate(tree.children[2], sidelist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
			del sidelist[evaluate(tree.children[0], looped = looped, func = func).name]
			merge(sidelist, symlist)
		else:
			sidelist = deepcopy(symlist)
			evaluate(tree.children[0], sidelist, looped = looped, func = func)
			while not tree.children[1].children or hardeval(tree.children[1], sidelist, looped = looped, func = func):
				result = evaluate(tree.children[-1], sidelist, looped = looped, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
				if len(tree.children) >= 4: evaluate(tree.children[2], sidelist, looped = looped, func = func)
	elif tree.token.content == 'try':
		try:
			_hardeval(tree.children[0], symlist, looped = looped, func = func)
		except Exception as e:
			sidelist = deepcopy(symlist)
			assign(evaluate(tree.children[1], sidelist), e, looped = looped, func = func)
			evaluate(tree.children[2], sidelist, looped = looped, func = func)
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
	elif tree.token.content == 'import':
		_import(tree.children[0], symlist)
	elif tree.token.content in ['exist', 'exists']:
		return exists(tree.children[0], symlist)
	elif tree.token.content in ['exist not', 'exists not']:
		return not exists(tree.children[0], symlist)
	elif 'break_statement' in treetype:
		if looped:
			return Statement('break')
		else:
			raise SyntaxError('break outside of loop')
	elif 'continue_statement' in treetype:
		if looped:
			return Statement('continue')
		else:
			raise SyntaxError('continue outside of loop')
	elif 'comma_expr' in treetype:
		return comma_mode([evaluate(child, symlist, looped = looped, func = func) for child in tree.children])
	elif 'bracket_expr' in treetype:
		if 'opfunc' in treetype:
			return Function(lambda *args, **kwargs: infix_operators[tree.children[0].token.content](*(args + (symlist,)), **kwargs), False)
		if 'bracket' in treetype:
				return evaluate(tree.children[0], symlist, tuple, looped = looped, func = func)
		elif 'list' in treetype:
			if tree.children:
				result = evaluate(tree.children[0], symlist, list, looped = looped, func = func)
				if not isinstance(result, list): result = [result]
				return result
			return []
		elif 'codeblock' in treetype:
			sidelist = deepcopy(symlist)
			for statement in tree.children:
				result = evaluate(statement, sidelist, looped = looped, func = func)
				if isinstance(result, Statement):
					merge(sidelist, symlist)
					return result
			merge(sidelist, symlist)
	elif tree.token.type == 'ternary':
		return evaluate(tree.children[1], symlist, looped = looped, func = func) if hardeval(tree.children[0], symlist, looped = looped, func = func) else evaluate(tree.children[2], symlist, looped = looped, func = func)
	elif 'call' in treetype:
		if tree.children[0].token.content in ['eval']:
			return hardeval(tree.children[0], symlist, looped = looped, func = func)(*([symlist] + [evaluate(child, symlist, looped = looped, func = func) for child in tree.children[1:]]))
		else:
			return hardeval(tree.children[0], symlist, looped = looped, func = func)(*[evaluate(child, symlist, looped = looped, func = func) for child in tree.children[1:]])
	elif 'getitem' in treetype:
		def getter():
			return hardeval(tree.children[0], symlist, looped = looped, func = func)[hardeval(tree.children[1], symlist, looped = looped, func = func)]
		def setter(*args):
			hardeval(tree.children[0], symlist, looped = looped, func = func)[hardeval(tree.children[1], symlist, looped = looped, func = func)] = args[0]
		return Identifier('.getitem', getter, setter)
	elif 'expression' in treetype:
		return evaluate(tree.children[0], symlist, looped = looped, func = func)

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
