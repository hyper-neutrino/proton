import proton_parser, lexer, operator, math, sys, builtins, time

from utils import *
from copy import *

path = ''

def setPath(newpath):
	global path
	path = newpath

def include(name):
	filename = ((path) and (path + '/')) + name + '.proton'
	with open(filename, 'r') as f:
		return complete(proton_parser.parse(lexer.tokenize(f.read())))[1]

def __include__(value, symlist, includer):
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
			if ('!' + module.name) in symlist:
				value = symlist[module.name]
			else:
				value = includer(module.name)
				symlist['!' + module.name] = value
			symlist[assign.name] = getattr(value, name.name)
		else:
			name, assign = evaluate(value.children[0], symlist), evaluate(value.children[1], symlist)
			assert isinstance(name, Identifier) and isinstance(assign, Identifier)
			symlist[assign.name] = includer(name.name)
	elif 'expression' in treetype and 'keyword' in tokentype and value.token.content == 'from':
		name = evaluate(value.children[0], symlist)
		module = evaluate(value.children[1], symlist)
		assert isinstance(name, Identifier) and isinstance(module, Identifier)
		if ('!' + module.name) in symlist:
			value = symlist[module.name]
		else:
			value = includer(module.name)
			symlist['!' + module.name] = value
		symlist[name.name] = getattr(value, name.name)
	elif 'identifier' in tokentype:
		value = evaluate(value, symlist)
		symlist[value.name] = includer(value.name)
	else:
		raise SyntaxError('import statement not recognized')

def _import(value, symlist):
	__include__(value, symlist, __import__)

def clone_scope(scope):
	return {x: scope[x] for x in scope}

def delete(item, symlist):
	if isinstance(item, (list, tuple, set)):
		for i in item:
			delete(i, symlist)
	elif isinstance(item, Identifier):
		del symlist[item.name]
	else:
		raise SyntaxError('del can only be used on identifiers or lists of identifiers')

def assign_method_parameters(parameters, expression, scope):
	if 'comma_expr' in expression.type.split('/'):
		pairs = []
		varargs = False
		paramindex = 0
		required = [1 if child.token.type == 'unifix_operator' and child.token.content == '*' else 2 if child.type == 'nullable/expression' else 0 for child in expression.children]
		extra = len(parameters) - required.count(0)
		for i in range(len(required)):
			if required[i] == 2 and extra > 0:
				required[i] = 0
				extra -= 1
		if extra > 0 and not required.count(1):
			raise RuntimeError('Too many arguments given')
		for index in range(len(expression.children)):
			if required[index] == 2: continue
			if paramindex >= len(parameters):
				raise IndexError('Not enough arguments given')
			subexpr = expression.children[index]
			if subexpr.token.type == 'unifix_operator' and subexpr.token.content == '*':
				if varargs:
					raise RuntimeError('There can only be one varargs expression in a function declaration')
				else:
					varargs = True
					end = len(parameters) - len(expression.children) + index + 1
					scope[evaluate(subexpr.children[0], scope).name] = tuple(parameters[paramindex:end])
					paramindex = end
			else:
				scope[evaluate(subexpr).name] = parameters[paramindex]
				paramindex += 1
	elif 'bracket_expr' in expression.type.split('/'):
		if expression.children:
			assign_method_parameters(parameters, expression.children[0], scope)
	else:
		if expression.token.type == 'unifix_operator' and expression.token.content == '*':
			scope[evaluate(expression.children[0]).name] = tuple(parameters)
		elif expression.token.type == 'nullable/expression':
			name = evaluate(expression.children[0].name)
			if parameters:
				scope[name] = parameters[0]
			else:
				scope[name] = 0
				del scope[name]
		else:
			scope[evaluate(expression).name] = parameters[0]

def _s(op):
	def inner(x, y):
		return safechain([lambda: op(x, y), lambda: op(x, f(y)), lambda: op(f(x), y), lambda: op(f(x), f(y))])
	return inner

def _(op, l = True, r = True):
	L = f if l else lambda k: k
	R = f if r else lambda k: k
	return lambda x, y, z: _s(op)(L(evaluate(x, z)), R(evaluate(y, z)))

def __(op):
	return lambda x, y, z: (lambda k, j: k(_s(op)(k, j)))(evaluate(x, z), f(evaluate(y, z)))

def subref(x, y, z):
	x, y = f(evaluate(x, z)), evaluate(y, z)
	ident = isinstance(y, Identifier)
	y = y.name if ident else y
	def getter():
		return getattr(x, y) if ident and hasattr(x, y) else x[y]
	def setter(*args):
		if ident and hasattr(x, y):
			setattr(x, y, args[0])
		else:
			x[y] = args[0]
	return Identifier('', getter, setter)

def walk(x):
	if hasattr(x, '__iter__') and not isinstance(x, str):
		result = []
		for y in x:
			result += walk(y)
		return result
	return [x]

def assign(x, y, r = False):
	if not all(isinstance(node, Identifier) for node in walk(x)):
		if r:
			raise RuntimeError('Cannot assign to non-identifier')
		else:
			return assign(y, x, True)
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

def true_inverse(function):
	return lambda *args, **kwargs: function(*(args + (True,)), **kwargs)

def english_convenienced_function(operand):
	def inner(x, y, s):
		if isinstance(x, proton_parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'and':
			return     (inner(x.children[0], y, s) and inner(x.children[1], y, s))
		elif isinstance(x, proton_parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'or':
			return     (inner(x.children[0], y, s) or inner(x.children[1], y, s))
		elif isinstance(x, proton_parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'nand':
			return not (inner(x.children[0], y, s) and inner(x.children[1], y, s))
		elif isinstance(x, proton_parser.ASTNode) and x.token.type == 'binary_operator' and x.token.content == 'nor':
			return not (inner(x.children[0], y, s) or inner(x.children[1], y, s))
		elif isinstance(y, proton_parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'and':
			return     (inner(x, y.children[0], s) and inner(x, y.children[1], s))
		elif isinstance(y, proton_parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'or':
			return     (inner(x, y.children[0], s) or inner(x, y.children[1], s))
		elif isinstance(y, proton_parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'nand':
			return not (inner(x, y.children[0], s) and inner(x, y.children[1], s))
		elif isinstance(y, proton_parser.ASTNode) and y.token.type == 'binary_operator' and y.token.content == 'nor':
			return not (inner(x, y.children[0], s) or inner(x, y.children[1], s))
		else:
			return operand(f(evaluate(x, s)), f(evaluate(y, s)))
	return inner

def EQ(x, y):
	return f(x) == f(y)

def instanceof(x, y):
	if isinstance(x, dict) and '__type__' in x:
		return x['__type__']().function == y
	else:
		if type(x) == y:
			return True
		if type(y) == type:
			return isinstance(x, y)
		return False

def deep_search(x, y):
	if x == y or hasattr(y, '__iter__') and x in y: return True
	if hasattr(y, '__iter__'):
		for Y in y:
			if deep_search(x, Y):
				return True
	return False

def repeat(x, y):
	if hasattr(y, '__iter__'): x, y = y, x
	result = []
	index = 0
	for i in range(int(len(x) * y)):
		result.append(x[index])
		index += 1
		index %= len(x)
	return ''.join(result) if isinstance(x, str) else type(x)(result)

def s_(function, modif):
	return lambda *args: function(*map(modif, args))

infix_operators = {
	'**': _(operator.pow),
	'>>': _(s_(operator.rshift, int)),
	'<<': _(s_(operator.lshift, int)),
	'*': _(lambda x, y: repeat(x, y) if hasattr(x, '__iter__') ^ hasattr(y, '__iter__') else x * y),
	'/': _(operator.truediv),
	'//': _(operator.floordiv),
	'||': _(lambda x, y: y % x == 0),
	'%': _(operator.mod),
	'+': _(lambda x, y: x + str(y) if isinstance(x, str) else str(x) + y if isinstance(y, str) else x + y),
	'-': _(operator.sub),
	'>': _(operator.gt),
	'<': _(operator.lt),
	'<=': _(operator.le),
	'>=': _(operator.ge),
	'==': _(EQ),
	'!=': _(inverse(EQ)),
	'..': _(lambda x, y: list(range(x, y))),
	'in': english_convenienced_function(lambda x, y: x in y),
	'not in': english_convenienced_function(lambda x, y: x not in y),
	'inside': english_convenienced_function(deep_search),
	'not inside': english_convenienced_function(inverse(deep_search)),
	'is': english_convenienced_function(lambda x, y: instanceof(x, y.function)),
	'is not': english_convenienced_function(inverse(lambda x, y: instanceof(x, y.function))),
	'are': english_convenienced_function(lambda x, y: instanceof(x, y.function)),
	'are not': english_convenienced_function(inverse(lambda x, y: instanceof(x, y.function))),
	'&': _(s_(operator.and_, int), False, False),
	'|': _(s_(operator.or_, int)),
	'^': _(s_(operator.xor, int)),
	'and'  : lambda x, y, z: (lambda k: k if not k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'or'   : lambda x, y, z: (lambda k: k   if   k else f(evaluate(y, z)))(f(evaluate(x, z))),
	'nand' : lambda x, y, z: (lambda k: f(evaluate(y, z))   if   k else k)(f(evaluate(x, z))),
	'nor'  : lambda x, y, z: (lambda k: f(evaluate(y, z)) if not k else k)(f(evaluate(x, z))),
	'**=': __(operator.pow),
	'*=': __(operator.mul),
	'/=': __(operator.truediv),
	'//=': __(operator.floordiv),
	'+=': __(operator.add),
	'-=': __(operator.sub),
	'>>=': __(s_(operator.rshift, int)),
	'<<=': __(s_(operator.lshift, int)),
	'%=': __(operator.mod),
	'&=': __(s_(operator.and_, int)),
	'|=': __(s_(operator.or_, int)),
	'&&=': __(lambda x, y: f(x) and y),
	'||=': __(lambda x, y: f(x) or y),
	'=': lambda x, y, z: assign(evaluate(x, z), evaluate(y, z)),
	':=': lambda x, y, z: assign(evaluate(x, z), deepcopy(hardeval(y, z))),
	'.': subref
}

def ___(op):
	return lambda x, z: op(f(evaluate(x, z)))

def exists(value, symlist):
	if value.token.type == 'binary_operator' and value.token.content == 'and':
		return     (exists(value.children[0], symlist) and exists(value.children[1], symlist))
	elif value.token.type == 'binary_operator' and value.token.content == 'or':
		return     (exists(value.children[0], symlist)  or exists(value.children[1], symlist))
	if value.token.type == 'binary_operator' and value.token.content == 'nand':
		return not (exists(value.children[0], symlist) and exists(value.children[1], symlist))
	elif value.token.type == 'binary_operator' and value.token.content == 'nor':
		return not (exists(value.children[0], symlist)  or exists(value.children[1], symlist))
	elif 'bracket' in value.type and 'bracket_expr' in value.type:
		return exists(value.children[0], symlist)
	else:
		ident = evaluate(value, symlist)
		if isinstance(ident, Identifier):
			return ident.name in symlist or ident.name == '.getitem' and cando(ident) or ident() is not None
		return True

def negative(value):
	return -value

def elapsed(gen):
	start = time.time()
	gen()
	return time.time() - start

def get_index(array, index):
	if isinstance(array, int):
		return int(bool(array & (1 << index)))
	else:
		return array[index]

def set_index(reference, array, index, value):
	if isinstance(array, int):
		if get_index(array, index) and not value:
			array -= array & (1 << index)
		elif not get_index(array, index) and value:
			array += array & (1 << index)
		return reference(array)
	elif isinstance(array, str):
		return reference(array[:index] + value + array[index + 1:])
	else:
		array[index] = value
		return array

def get_indices(array, indices):
	if isinstance(array, int):
		result = 0
		for index in indices:
			result |= (array & (1 << index))
		return result
	else:
		if isinstance(array, str):
			return ''.join(array[index] for index in indices)
		else:
			return type(array)(array[index] for index in indices)

class DEL:
	def __init__(self):
		pass
	def __eq__(self, other):
		return isinstance(other, DEL)
	def __ne__(self, other):
		return not isinstance(other, DEL)

def set_indices(reference, array, indices, values):
	for index, value in zip(indices, list(values)):
		array = set_index(reference, array, index, value)
	s = isinstance(array, str)
	if s: array = list(array)
	for index in indices[len(values):]:
		array = set_index(reference, array, index, DEL())
	if isinstance(reference, Identifier):
		if s:
			return reference(''.join(filter(DEL().__ne__, array)))
		else:
			return reference(type(array)(filter(DEL().__ne__, array)))

def autosplat(function):
	def inner(*args, **kwargs):
		result = []
		for arg in args:
			if isinstance(arg, (list, tuple, set)):
				result += list(arg)
			else:
				result += [arg]
		return function(*result)
	return getfunction(inner)

prefix_operators = {
	'!': ___(operator.not_),
	'~': ___(s_(operator.invert)),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)),
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z)),
	'-': ___(negative),
	'@': ___(lambda x: x.setCaching(True)),
	'!!': ___(lambda x: x.setPartial(True)),
	'&': ___(lambda x: getfunction(lambda *a, **k: call(x, a, **k))),
	'$': ___(lambda x: x.wipeCache()),
	'$$': ___(lambda x: x.setCaching(False).wipeCache()),
	'timeof': lambda x, s: elapsed(lambda: evaluate(x, s)),
	'sizeof': ___(sys.getsizeof),
	'*': ___(lambda x: autosplat(x) if isinstance(x, Function) else splat(x)),
	'%%': ___(lambda x: Function(lambda *args, **kwargs: splat(x(*args, **kwargs))))
}

postfix_operators = {
	'!': ___(lambda x: type(x)(math.gamma(x + 1))),
	'++': lambda x, z: (lambda k: k(f(k) + 1))(evaluate(x, z)) - 1,
	'--': lambda x, z: (lambda k: k(f(k) - 1))(evaluate(x, z)) + 1
}

class Statement:
	def __init__(self, name, value = None):
		self.name = name
		self.value = value

class Identifier:
	def __init__(self, name, getter, setter):
		self.name = name
		self.getter = getter
		self.setter = setter
	def __call__(self, *args):
		if args: self.setter(*args)
		return self.getter()

class MapExpr:
	def __init__(self, key, value):
		self.key = key
		self.value = value

setident(Identifier)

def listify(func):
	return lambda *args, **kwargs: list(func(*args, **kwargs))

default = {

}

for name in dir(builtins):
	try:
		default[name] = eval(name)
		if isinstance(default[name], (type(print), type(lambda:0), type(type))):
			default[name] = getfunction(default[name])
	except:
		pass
	
default["int"] = getfunction(sympy.Integer)
default["float"] = getfunction(sympy.Rational)
# TODO default["complex"]

class proton_str(str):
	def __init__(self, obj):
		if isinstance(obj, dict):
			self = '[%s]' % ', '.join(sorted([ascii(key) + ' :> ' + ascii(obj[key]) for key in obj]))
		elif isinstance(obj, Function):
			self = proton_str(obj.function)
		else:
			self = str(obj)

default['eval'] = getfunction(lambda s, x: complete(proton_parser.parse(lexer.tokenize(x)), s))
default['Function'] = Function
default['str'] = proton_str
default['type'] = lambda x: x['__type__'] if isinstance(x, dict) and '__type__' in x else type(x)
default['class'] = make_type

def merge(d1, d2):
	for x in d1:
		if x in d2 or x.startswith('!'):
			if d1[x] != d2[x]:
				d2[x] = d1[x]

def indices(array, ref, symlist):
	reftype = ref.type.split('/')
	tokentype = ref.token.type.replace(':', '/').split('/')
	if 'comma_expr' in reftype:
		return sum([indices(array, subref, symlist) for subref in ref.children], [])
	elif 'slice' in reftype:
		nones = [x is None for x in ref.children]
		while len(nones) < 3: nones.append(True)
		id = nones[0] * 4 + nones[1] * 2 + nones[2] * 1
		val = lambda i: hardeval(ref.children[i], symlist)
		I = list(range(len(array)))
		if id == 0:
			return I[val(0):val(1):val(2)]
		elif id == 1:
			return I[val(0):val(1)]
		elif id == 2:
			return I[val(0)::val(2)]
		elif id == 3:
			return I[val(0):]
		elif id == 4:
			return I[:val(1):val(2)]
		elif id == 5:
			return I[:val(1)]
		elif id == 6:
			return I[::val(2)]
		elif id == 7:
			return I[:]
	else:
		return [hardeval(ref, symlist)]

def hardeval(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	return f(evaluate(tree, symlist, comma_mode, looped, func))

def _hardeval(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	return f(_evaluate(tree, symlist, comma_mode, looped, func))

def _evaluate(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	sidelist = clone_scope(symlist)
	value = evaluate(tree, sidelist, comma_mode, looped, func)
	merge(sidelist, symlist)
	return value

def evaluate(tree, symlist = None, comma_mode = tuple, looped = False, func = False):
	if not isinstance(tree, proton_parser.ASTNode):
		return tree
	setevalp(lambda k: complete(proton_parser.parse(lexer.tokenize(k)), symlist = symlist or default)[0])
	symlist = symlist or default
	treetype = tree.type.split('/')
	tokentype = tree.token.type.replace(':', '/').split('/')
	if 'literal' in tokentype:
		return h(tree.token.content)
	elif 'identifier' in tokentype:
		def getter():
			return None if tree.token.content not in symlist else symlist[tree.token.content]
		def setter(*args):
			if 'const' in symlist and tree.token.content in symlist['const']:
				raise RuntimeError('Cannot set constant value ' + tree.token.content)
			symlist[tree.token.content] = args[0]
		return Identifier(tree.token.content, getter, setter)
	elif 'const' in treetype:
		left  = evaluate(tree.children[0], symlist, looped = looped, func = func)
		right = evaluate(tree.children[1], symlist, looped = looped, func = func)
		if not all(isinstance(node, Identifier) for node in walk(left)):
			left, right = right, left
		assign(left, right)
		if 'const' not in symlist: symlist['const'] = []
		for node in walk(left):
			symlist['const'] += node.name
	elif 'comp' in treetype:
		result = []
		for val in hardeval(tree.children[2], symlist, looped = looped, func = func):
			assign(evaluate(tree.children[1], symlist, looped = looped, func = func), val)
			if len(tree.children) < 4 or hardeval(tree.children[3], symlist, looped = looped, func = func):
				result.append(_hardeval(tree.children[0], symlist, looped = looped, func = func))
		return result
	elif tree.token.content == 'unless':
		if hardeval(tree.children[1], symlist, looped = looped, func = func):
			return _evaluate(tree.children[2], symlist, looped = looped, func = func)
		else:
			return _evaluate(tree.children[0], symlist, looped = looped, func = func)
	elif tree.token.content == 'if':
		if hardeval(tree.children[0], symlist, looped = looped, func = func):
			return _evaluate(tree.children[1], symlist, looped = looped, func = func)
		elif len(tree.children) > 2:
			return _evaluate(tree.children[2], symlist, looped = looped, func = func)
	elif tree.token.content == 'while':
		if 'whileas' in treetype:
			sidelist = clone_scope(symlist)
			value = hardeval(tree.children[0], sidelist, looped = looped, func = func)
			result = None
			while value:
				assign(evaluate(tree.children[1], sidelist, looped = looped, func = func), value)
				result = evaluate(tree.children[2], sidelist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
					if result.name == 'return': return result
				value = hardeval(tree.children[0], sidelist, looped = looped, func = func)
			else:
				if len(tree.children) > 3:
					evaluate(tree.children[3])
			del sidelist[evaluate(tree.children[1], looped = looped, func = func).name]
			merge(sidelist, symlist)
			return result
		else:
			result = None
			while hardeval(tree.children[0], symlist, looped = looped, func = func):
				result = _evaluate(tree.children[1], symlist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
					if result.name == 'return': return result
			else:
				if len(tree.children) > 2:
					evaluate(tree.children[2])
			return result
	elif tree.token.content == 'repeat':
		if 'repeatinto' in treetype:
			sidelist = clone_scope(symlist)
			value = f(hardeval(tree.children[0], sidelist, looped = looped, func = func))
			index = 0
			result = None
			while index < value:
				assign(evaluate(tree.children[1], sidelist, looped = looped, func = func), index)
				result = evaluate(tree.children[2], sidelist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
					if result.name == 'return': return result
				index += 1
			del sidelist[evaluate(tree.children[1], looped = looped, func = func).name]
			merge(sidelist, symlist)
			return result
		else:
			result = None
			value = f(hardeval(tree.children[0], symlist, looped = looped, func = func))
			while value:
				result = _evaluate(tree.children[1], symlist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
					if result.name == 'return': return result
				value -= 1
			return result
	elif tree.token.content == 'for':
		if 'foreach' in treetype:
			sidelist = clone_scope(symlist)
			result = None
			for val in hardeval(tree.children[1], sidelist, looped = looped, func = func):
				assign(evaluate(tree.children[0], sidelist, looped = looped, func = func), val)
				result = evaluate(tree.children[2], sidelist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': break
					if result.name == 'continue': continue
					if result.name == 'return':
						try:
							del sidelist[evaluate(tree.children[0], looped = looped, func = func).name]
						except: pass
						merge(sidelist, symlist)
						return result
			else:
				if len(tree.children) > 3:
					evaluate(tree.children[3])
			try:
				del sidelist[evaluate(tree.children[0], looped = looped, func = func).name]
			except: pass
			merge(sidelist, symlist)
			return result
		else:
			sidelist = clone_scope(symlist)
			evaluate(tree.children[0], sidelist, looped = looped, func = func)
			result = None
			broken = False
			while not tree.children[1].children or hardeval(tree.children[1], sidelist, looped = looped, func = func):
				result = evaluate(tree.children[-1], sidelist, looped = True, func = func)
				if isinstance(result, Statement):
					if result.name == 'break': broken = True; break
					if result.name == 'continue': continue
					if result.name == 'return':
						try:
							del sidelist[evaluate(tree.children[0], looped = looped, func = func).name]
						except: pass
						merge(sidelist, symlist)
						return result
				if len(tree.children) >= 4: evaluate(tree.children[2], sidelist, looped = looped, func = func)
			if not broken:
				if len(tree.children) > 3:
					evaluate(tree.children[3])
			merge(sidelist, symlist)
			return result
	elif tree.token.content == 'try':
		try:
			return _hardeval(tree.children[0], symlist, looped = looped, func = func)
		except Exception as e:
			sidelist = clone_scope(symlist)
			assign(evaluate(tree.children[1], sidelist), e, looped = looped, func = func)
			return evaluate(tree.children[2], sidelist, looped = looped, func = func)
	elif 'prebinopfunc' in treetype:
		def inner(*args, **kwargs):
			arguments = args[:1] + (evaluate(tree.children[0], symlist, looped = looped, func = func), symlist)
			value = infix_operators[tree.token.content](*arguments)
			return args[1:] and (value,) + args[1:] or value
		return inner
	elif 'postbinopfunc' in treetype:
		def inner(*args, **kwargs):
			arguments = (evaluate(tree.children[0], symlist, looped = looped, func = func),) + args[:1] + (symlist,)
			value = infix_operators[tree.token.content](*arguments)
			return args[1:] and (value,) + args[1:] or value
		return inner
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
	elif tree.token.content == 'include':
		__include__(tree.children[0], symlist, include)
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
	elif 'return_statement' in treetype:
		if func:
			return Statement('return', f(evaluate(tree.children[0], symlist)))
		else:
			raise SyntaxError('return outside of function')
	elif 'comma_expr' in treetype:
		result = comma_mode(evaluate(child, symlist, looped = looped, func = func) for child in tree.children)
		def flatten(array):
			result = comma_mode()
			if isinstance(array, (list, tuple, set)):
				for element in array: result += flatten(element)
			else:
				result = comma_mode([array])
			return result
		if all(isinstance(f(element), Function) for element in flatten(result)):
			return getfunction(lambda *args, **kwargs: call(f(result), *args, **kwargs))
		return result
	elif 'mapping' in treetype:
		return MapExpr(*[evaluate(child, symlist, looped = looped, func = func) for child in tree.children])
	elif 'binopfunc' in treetype:
		def inner(*args, **kwargs):
			arguments = args[:2] + (symlist,)
			value = infix_operators[tree.token.content](*arguments)
			return args[2:] and (value,) + args[2:] or value
		return getfunction(inner, False)
	elif 'unopfunc' in treetype:
		def inner(*args, **kwargs):
			arguments = args[:2] + (symlist,)
			value = prefix_operators[tree.token.content](*arguments)
			return args[1:] and (value,) + args[1:] or value
		return getfunction(inner, False)
	elif 'bracket_expr' in treetype:
		if 'bracket' in treetype:
				return evaluate(tree.children[0], symlist, tuple, looped = looped, func = func)
		elif 'list' in treetype:
			if tree.children:
				result = evaluate(tree.children[0], symlist, list, looped = looped, func = func)
				innertype = tree.children[0].type.split('/')
				if 'comma_expr' not in innertype and 'comp' not in innertype: result = [result]
				maps = [isinstance(element, MapExpr) for element in result]
				if any(maps):
					if not all(maps):
						raise SyntaxError('Mixed mappings and non-mappings in list expression')
					return dict([(element.key, element.value) for element in result])
				return result
			return []
		elif 'codeblock' in treetype:
			sidelist = clone_scope(symlist)
			result = None
			for statement in tree.children:
				result = evaluate(statement, sidelist, looped = looped, func = True)
				if isinstance(result, Statement):
					merge(sidelist, symlist)
					if result.name == 'return': return result
					return result
			merge(sidelist, symlist)
			return result
	elif tree.token.type == 'ternary':
		return evaluate(tree.children[1], symlist, looped = looped, func = func) if hardeval(tree.children[0], symlist, looped = looped, func = func) else evaluate(tree.children[2], symlist, looped = looped, func = func)
	elif 'call' in treetype:
		if tree.children[0].token.content in ['eval']:
			return symlist['eval'](*([symlist] + [evaluate(child, symlist, looped = looped, func = func) for child in tree.children[1:]]))[0]
		else:
			args = []
			for argument in tree.children[1:]:
				if argument.token.type == 'unifix_operator' and argument.token.content == '*':
					for arg in f(hardeval(argument.children[0], symlist, looped = looped, func = func)):
						args.append(arg)
				else:
					args.append(evaluate(argument, symlist, looped = looped, func = func))
			func = f(evaluate(tree.children[0], symlist, looped = looped, func = func))
			return call(func, *args)
	elif 'getitem' in treetype:
		if 'comma_expr' in tree.children[1].type.split('/') or 'slice' in tree.children[1].type.split('/'):
			ref = evaluate(tree.children[0], symlist, looped = looped, func = func)
			array = f(ref)
			index_list = indices(array, tree.children[1], symlist)
			def getter():
				return get_indices(array, index_list)
			def setter(*args):
				set_indices(ref, array, index_list, args[0])
			return Identifier('.getitem', getter, setter)
		else:
			ref = evaluate(tree.children[0], symlist, looped = looped, func = func)
			array = f(ref)
			index = f(hardeval(tree.children[1], symlist))
			def getter():
				return get_index(array, index)
			def setter(*args):
				set_index(ref, array, index, args[0])
			return Identifier('.getitem', getter, setter)
	elif 'anonfunc' in treetype:
		def inner(*values):
			scope = clone_scope(symlist)
			try:
				assign_method_parameters(values, tree.children[0], scope)
				value = f(hardeval(tree.children[1], scope, func = True))
				merge(scope, symlist)
				if isinstance(value, Statement):
					if value.name == 'return': return value.value
					raise SystemExit('huh?')
				return value
			except IndexError:
				if getfunction(inner).partial:
					def _inner(*args):
						return inner(*(values + args))
					return _inner
				else:
					raise
		return getfunction(inner)
	elif 'del' in treetype:
		delete(evaluate(tree.children[0]), symlist)
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
		return (result, symlist)

def complete(tree, symlist = None):
	return Interpreter(tree).interpret(symlist)

def full(code, symlist = None):
	return complete(proton_parser.parse(lexer.tokenize(code)), symlist = symlist)
