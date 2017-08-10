import parser, lexer, operator, math, sys

def f(x):
	if isinstance(x, Identifier):
		return x()
	return x

def _(op):
	return lambda x, y: op(f(x), f(y))

def __(op):
	return lambda x, y: x(op(f(x), f(y)))

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
	'=': lambda x, y: x(f(y)),
}

def ___(op):
	return lambda x: op(f(x))

prefix_operators = {
	'!': ___(operator.not_)
}

postfix_operators = {
	'!': ___(lambda x: math.gamma(x + 1))
}

class Identifier:
	def __init__(self, getter, setter):
		self.getter = getter
		self.setter = setter
	def __call__(self, *args):
		if args: self.setter(*args)
		return self.getter()

def listify(func):
	return lambda *args, **kwargs: list(func(*args, **kwargs))

default = {
	'print': print,
	'eval': eval,
	'input': input,
	'range': listify(range),
	'sum': sum
}

def hardeval(tree, symlist = None, comma_mode = tuple):
	value = evaluate(tree, symlist, comma_mode)
	if isinstance(value, Identifier): return value()
	return value

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
			raise NameError("name '%s' is not defined" % tree.token.content)
		def setter(*args):
			symlist[tree.token.content] = args[0]
		return Identifier(getter, setter)
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
		return hardeval(tree.children[0], symlist)[hardeval(tree.children[1], symlist)]
	elif 'expression' in treetype:
		return evaluate(tree.children[0], symlist)

class Interpreter:
	def __init__(self, tree):
		self.tree = tree
	def interpret(self, symlist = None):
		symlist = symlist or {x: default[x] for x in default}
		for tree in self.tree:
			evaluate(tree, symlist)

def complete(tree, symlist = None):
	Interpreter(tree).interpret(symlist)
